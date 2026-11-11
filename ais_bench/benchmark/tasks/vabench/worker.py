
"""Isolated official model process, with an optional Ascend migration shim."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import importlib.metadata
import functools
import logging
import os
from pathlib import Path
import re
import sys
import traceback

from ais_bench.benchmark.utils.vabench.data import read_json, write_json
from .resources import submodules
from ais_bench.benchmark.summarizers.vabench.results import normalize_records


def initialize_device(device):
    import torch
    if device.startswith('npu'):
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError('Requested NPU is unavailable')
        # Only this child process sees the CUDA migration shim.
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
        torch.npu.set_device(device)
    elif device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable')
        torch.cuda.set_device(device)
    elif device != 'cpu':
        raise ValueError(f'Unsupported device {device}')
    return torch.device(device)


def _getattr_or_none(obj, name):
    """Best-effort lookup: dynamic __getattr__ may run, but errors stay isolated."""
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _observe_devices(module):
    """Inspect retained torch modules and ONNX sessions, without guessing."""
    devices, providers, seen = set(), set(), set()

    def walk(obj, depth=0):
        if id(obj) in seen or depth > 3:
            return
        seen.add(id(obj))
        # ctypes.LibraryLoader.__getattr__ tries to dlopen the attribute name;
        # a plain hasattr(obj, 'parameters') can therefore raise OSError.
        parameters = _getattr_or_none(obj, 'parameters')
        if callable(parameters):
            try:
                devices.update(str(p.device) for p in parameters())
            except Exception:
                pass
        get_providers = _getattr_or_none(obj, 'get_providers')
        if callable(get_providers):
            try:
                providers.update(get_providers())
            except Exception:
                pass
        if isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value, depth + 1)
        elif isinstance(obj, dict):
            for value in obj.values():
                walk(value, depth + 1)
        elif not isinstance(obj, type):
            try:
                attrs = vars(obj)
            except TypeError:
                attrs = None
            if attrs is not None:
                for name, value in attrs.items():
                    if not name.startswith('__'):
                        walk(value, depth + 1)
    for name, obj in vars(module).items():
        if 'MODEL' in name.upper() or 'CACHE' in name.upper():
            walk(obj)
    return dict(observed_torch_devices=sorted(devices), observed_onnx_providers=sorted(providers))


def _rebind_stream_handlers(old_stream, new_stream):
    """Move logging handlers off a redirect target that is about to close.

    A library that calls logging.basicConfig() while stderr is redirected
    leaves a handler pointing at the per-dimension backend.log. Logging to it
    after the file closes prints a traceback, which used to be misread as a
    metric failure.
    """
    loggers = [logging.getLogger()]
    loggers.extend(obj for obj in logging.Logger.manager.loggerDict.values()
                   if isinstance(obj, logging.Logger))
    for logger in loggers:
        for handler in list(logger.handlers):
            if _getattr_or_none(handler, 'stream') is old_stream:
                handler.acquire()
                try:
                    handler.stream = new_stream
                finally:
                    handler.release()


def instrument_models(module, observations, device):
    """Observe model factories before official cleanup drops their objects."""
    from types import SimpleNamespace
    if module.__name__.endswith('first_audiobox'):
        original_predictor = module.initialize_predictor
        def predictor_on_device(*args, **kwargs):
            predictor = original_predictor(*args, **kwargs)
            # The upstream factory chooses CUDA automatically and ignores
            # compute_*'s device. Explicit placement is essential for CPU.
            import torch
            predictor.device = torch.device(device)
            predictor.model.to(predictor.device)
            return predictor
        if not getattr(original_predictor, '_vabench_observer', False):
            predictor_on_device._vabench_observer = True
            module.initialize_predictor = predictor_on_device
    if module.__name__.endswith('second_lsa') and not device.startswith('cuda'):
        # InsightFace is an ONNX subcomponent; explicitly use CPU on A3.
        # Preserve the official detector models, det_size and allowed modules.
        face = importlib.import_module('third_party.syncnet_eval.latentsync.utils.face_detector')
        def initialize_face(self, device='cpu'):
            self.app = face.FaceAnalysis(
                allowed_modules=['detection', 'landmark_2d_106'],
                root=face._INSIGHTFACE_ROOT, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=(face.INSIGHTFACE_DETECT_SIZE,
                                                face.INSIGHTFACE_DETECT_SIZE))
            providers = set()
            for model in getattr(self.app, 'models', {}).values():
                session = getattr(model, 'session', None)
                if session is not None:
                    providers.update(session.get_providers())
            observations.append(dict(observed_torch_devices=[],
                                     observed_onnx_providers=sorted(providers)))
        face.FaceDetector.__init__ = initialize_face
        observations.append(dict(component='InsightFace', configured_device='cpu',
                                 reason='Explicit ONNX CPU subcomponent on non-CUDA runtime'))
    for name, func in list(vars(module).items()):
        if callable(func) and name.startswith(('_get_', '_load_')) and not getattr(func, '_vabench_observer', False):
            @functools.wraps(func)
            def wrapped(*args, _func=func, **kwargs):
                result = _func(*args, **kwargs)
                if module.__name__.endswith('first_dnsmos') and device == 'cpu':
                    # ONNX execution providers are independent of torch.device.
                    for value in vars(result).values() if hasattr(result, '__dict__') else []:
                        if hasattr(value, 'set_providers'):
                            value.set_providers(['CPUExecutionProvider'])
                observations.append(_observe_devices(SimpleNamespace(_MODEL=result)))
                return result
            wrapped._vabench_observer = True
            setattr(module, name, wrapped)


def has_backend_error(log_text):
    return bool(re.search(
        r'Traceback \(most recent|OOM_ERROR|out of memory|'
        r'(?:^|\])\s*(?:Error\b|Failed\b|\[.*?\]\s*Error\b)|'
        r'FFmpeg failed for |SoundFile read failed for |'
        r'处理音频文件失败|视频评估失败', log_text, re.I | re.M))


def failed_sample_ids(log_text):
    """Samples named by official per-video error messages.

    VABench prints ``Error processing <path>`` (CLAP, Desync) or
    ``Error processing sample <path>`` (ImageBind) and then keeps looping.
    The remaining results in that scene stay valid.
    """
    return {Path(match.group(1).strip()).stem for match in re.finditer(
        r'(?:Error processing (?:sample )?|FFmpeg failed for |SoundFile read failed for )'
        r'(.+?):(?:\s|$)', log_text, re.I)}


def attribute_backend_errors(log_text, expected):
    """Release scene-wide protection only when every error event is attributed.

    One named error does not explain unrelated decoder/global errors. A traceback
    immediately following a named event belongs to that event; a bare or second
    traceback without a new named event is conservatively unresolved.
    """
    known = {row['idx'] for row in expected}
    failed, unresolved = set(), []
    current_named, traceback_seen = False, False
    for line in log_text.splitlines():
        named = failed_sample_ids(line)
        if named:
            matched = named & known
            failed.update(matched)
            current_named, traceback_seen = bool(matched), False
            if named - known:
                unresolved.append(line)
        elif 'Traceback (most recent' in line:
            if not current_named or traceback_seen:
                unresolved.append(line)
            traceback_seen = True
        elif has_backend_error(line):
            unresolved.append(line)
            current_named, traceback_seen = False, False
        elif line.strip():
            # An intervening progress/success message breaks traceback adjacency.
            current_named = False
    return dict(failed_ids=failed, unresolved=unresolved,
                scene_unverified=bool(unresolved))


def execute(request):
    root = Path(request['source_root']).resolve()
    os.environ['VABENCH_CACHE_DIR'] = request['cache_dir']
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(root))
    device = initialize_device(request['device'])
    if 'second_lsa' in request['dimensions']:
        # Official VABench permanently enables cuDNN deterministic mode when
        # Omni is loaded, and later LSA categories inherit that global setting.
        # LSA runs in its own worker here, so set the same backend mode before
        # importing the official module to keep native LSA scores aligned.
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    omni = importlib.import_module('vabench.omni_runtime')
    # Capture model residency when loaded, before official release_omni()
    # discards its model. Do not change processor, decoding, or model weights.
    observed_omni = set()
    original_get = omni.get_omni_model_and_processor

    def observe_omni(*args, **kwargs):
        model, processor = original_get(*args, **kwargs)
        observed_omni.update(str(p.device) for p in model.parameters())
        return model, processor

    omni.get_omni_model_and_processor = observe_omni
    grouped = request['grouped']
    expected = request['records']
    details = {}
    for dimension in request['dimensions']:
        records, errors, residency, factory_observations = [], [], [], []
        for category, metadata in grouped.items():
            scene_expected = [r for r in expected if r['category'] == category]
            sandbox = Path(request['work_dir']) / dimension / category
            sandbox.mkdir(parents=True, exist_ok=True)
            full_info = sandbox / 'full_info.json'
            write_json(full_info, metadata)
            old_cwd = Path.cwd()
            log_file = sandbox / 'backend.log'
            raw, observed, caught = [], {}, None
            log = None
            try:
                os.chdir(sandbox)
                with log_file.open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
                    module = importlib.import_module(f'vabench.{dimension}')
                    instrument_models(module, factory_observations, request['device'])
                    params = dict(request.get('params', {}))
                    params.update(request.get('dim_params', {}).get(dimension, {}))
                    if dimension.startswith(('third_', 'fourth_')):
                        params['qwen_omni_ckpt'] = str(Path(request['cache_dir']) / 'Qwen2.5_omni_7B')
                        params['local_files_only'] = True
                        if request['device'].startswith('npu'):
                            params['device_map'] = {'': request['device']}
                            params.setdefault('attn_implementation', 'eager')
                    result = getattr(module, f'compute_{dimension}')(
                        str(full_info), device, submodules(dimension, request['cache_dir']), **params)
                    raw = result[1]
                    observed = _observe_devices(module)
            except Exception:
                caught = traceback.format_exc()
                errors.append(dict(category=category, error=caught))
            finally:
                os.chdir(old_cwd)
                if log is not None:
                    # The with-block restored sys.stderr; detach handlers that
                    # were created while it pointed at this closed log file.
                    _rebind_stream_handlers(log, sys.stderr)
            text = log_file.read_text(encoding='utf-8') if log_file.exists() else ''
            bad_log = has_backend_error(text)
            attribution = attribute_backend_errors(text, scene_expected)
            named_failures = attribution['failed_ids']
            if bad_log and caught is None:
                errors.append(dict(category=category,
                                   error='Backend reported error; inspect backend.log',
                                   samples=sorted(named_failures),
                                   unresolved_errors=attribution['unresolved']))
            try:
                all_failed = bool(caught or attribution['scene_unverified']
                                  or (bad_log and not named_failures))
                records.extend(normalize_records(
                    dimension, raw, scene_expected, backend_errors=all_failed,
                    failed_ids=None if all_failed else named_failures))
            except Exception:
                errors.append(dict(category=category, error=traceback.format_exc()))
                records.extend(normalize_records(dimension, [], scene_expected, backend_errors=True))
            residency.append(dict(category=category, **observed))
        details[dimension] = dict(records=records, errors=errors,
                                  requested_device=request['device'], device_observations=residency,
                                  model_factory_observations=factory_observations)
        write_json(Path(request['work_dir']) / 'partial.json', details)
    for dim in details:
        if dim.startswith(('third_', 'fourth_')):
            details[dim]['observed_omni_devices'] = sorted(observed_omni)
    omni.release_omni()
    versions = {}
    for package in ('torch', 'torch-npu', 'transformers', 'onnxruntime', 'onnxruntime-gpu', 'qwen-omni-utils'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(details=details, environment=dict(python=sys.version, packages=versions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    args = parser.parse_args()
    request = read_json(args.request)
    try:
        result = execute(request)
        write_json(request['response'], result)
    except Exception:
        write_json(request['response'], dict(error=traceback.format_exc()))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())