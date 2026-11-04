"""Reset each owned vLLM instance before warmup; retries belong to the case."""
import json
import math
import re
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import urlsplit


def require_idle(metrics):
    for name in ('num_requests_running', 'num_requests_waiting'):
        values = [float(x) for x in re.findall(
            r'^vllm:' + name + r'(?:\{[^\n]*\})?\s+(\S+)', metrics, re.M)]
        if not values or any(not math.isfinite(v) or v != 0 for v in values):
            raise RuntimeError('Instance not verified idle: ' + name)


def reset_instances(folder, ports, http, target_urls=None):
    """http rejects non-2xx responses. Never route reset through the proxy."""
    report = {'status': 'FAILED', 'scope': 'local_prefix_cache',
              'time': datetime.now(timezone.utc).isoformat(), 'instances': []}
    try:
        if not ports or len(ports) != len(set(ports)):
            raise ValueError('Expected distinct model instance reset ports')
        # Check every instance before issuing any mutating request.
        for port in ports:
            item = {'target_id': port, 'url': (target_urls[port] if target_urls else f'http://127.0.0.1:{port}')+'/reset_prefix_cache',
                    'status': 'CHECKING_IDLE'}
            item['port']=urlsplit(item['url']).port
            report['instances'].append(item)
            metrics = http(port, '/metrics')
            (folder/f'reset-{port}-idle.txt').write_text(metrics)
            require_idle(metrics)
            item['status'] = 'IDLE'
        failures = []
        for item in report['instances']:
            port = item['target_id']
            try:
                body = http(port, '/reset_prefix_cache', 'POST')
                (folder/f'reset-{port}-0.json').write_text(body)
                response = json.loads(body)
                if not isinstance(response, dict) or response.get('success') is not True:
                    raise RuntimeError('Reset not confirmed; blocks may still be held')
                item['status'] = 'PASSED'
            except Exception as exc:
                item.update(status='FAILED', error=repr(exc))
                if isinstance(exc, HTTPError):
                    item['http_status'] = exc.code
                    (folder/f'reset-{port}-http-error.txt').write_text(exc.read().decode(errors='replace'))
                failures.append(port)
        if failures:
            raise RuntimeError('Prefix reset failed on instances: '+str(failures))
        report['status'] = 'PASSED'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        (folder/'cache-reset.json').write_text(json.dumps(report, indent=2))
