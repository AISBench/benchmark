import sys
import json
import asyncio
from concurrent.futures import Future
import multiprocessing as mp
import time
import unittest
from unittest import mock
from copy import deepcopy
import requests

from ais_bench.benchmark.models import BaseAPIModel, APITemplateParser
from ais_bench.benchmark.models.api_models.base_api import (
    AsyncPrefillLimiter,
    PrefillLease,
    SharedPrefillLimiter,
)
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchTypeError,
    AISBenchValueError,
    AISBenchRuntimeError,
    AISBenchKeyError,
)


class MockResponse:
    def __init__(self, status_code, json_data=None, text=None, reason=None):
        self.status_code = status_code
        self._json_data = json_data
        self._text = text
        self.reason = reason

    def json(self):
        return self._json_data

    def text(self):
        return self._text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP Error: {self.status_code}")


def _make_stream_context(chunks=(), status=200, reason=""):
    response = mock.MagicMock()
    response.status = status
    response.reason = reason

    async def content():
        for chunk in chunks:
            yield chunk

    response.content = content()
    context = mock.MagicMock()
    context.__aenter__ = mock.AsyncMock(return_value=response)
    context.__aexit__ = mock.AsyncMock(return_value=None)
    return context


def _prefill_process_worker(shared_limiter, start_event, iterations):
    if not start_event.wait(timeout=20):
        return
    for _ in range(iterations):
        if not shared_limiter.semaphore.acquire(timeout=10):
            raise RuntimeError("Timed out acquiring shared prefill permit")
        with shared_limiter.inflight.get_lock():
            shared_limiter.inflight.value += 1
            shared_limiter.max_inflight.value = max(
                shared_limiter.max_inflight.value,
                shared_limiter.inflight.value,
            )
        time.sleep(0.01)
        with shared_limiter.inflight.get_lock():
            shared_limiter.inflight.value -= 1
        shared_limiter.semaphore.release()


class TestBaseAPIModel(unittest.TestCase):
    def setUp(self):
        # 创建一个具体的BaseAPIModel子类用于测试
        class ConcreteAPIModel(BaseAPIModel):
            def _get_url(self):
                return self.base_url + "test"

            async def get_request_body(self, input_data, max_out_len, output, **args):
                return {"prompt": input_data, "max_tokens": max_out_len}

            async def parse_text_response(self, data, output):
                output.text = data.get("response", "")

            async def parse_stream_response(self, data, output):
                output.text += data.get("chunk", "")

        self.model_class = ConcreteAPIModel
        self.default_kwargs = {
            "path": "test-model",
            "stream": False,
            "max_out_len": 100,
            "retry": 1,
            "host_ip": "127.0.0.1",
            "host_port": 8000,
        }
        self.ipv6_kwargs = self.default_kwargs.copy()
        self.ipv6_kwargs["host_ip"] = "::1"

    def test_init(self):
        model = self.model_class(**self.default_kwargs)
        self.assertEqual(model.path, "test-model")
        self.assertEqual(model.stream, False)
        self.assertEqual(model.max_out_len, 100)
        self.assertEqual(model.retry, 1)
        self.assertEqual(model.base_url, "http://127.0.0.1:8000/")

    def test_init_with_ipv6_host_ip(self):
        model = self.model_class(**self.ipv6_kwargs)
        self.assertEqual(model.base_url, "http://[::1]:8000/")

    def test_init_with_url(self):
        kwargs = self.default_kwargs.copy()
        kwargs["url"] = "https://test-api.com/v1"
        model = self.model_class(**kwargs)
        self.assertEqual(model.base_url, "https://test-api.com/v1/")

        # Test that URL already with trailing slash remains unchanged
        kwargs["url"] = "https://test-api.com/v1/"
        model = self.model_class(**kwargs)
        self.assertEqual(model.base_url, "https://test-api.com/v1/")

        # Test that URL without protocol gets trailing slash
        kwargs["url"] = "test-api.com/path"
        model = self.model_class(**kwargs)
        self.assertEqual(model.base_url, "http://test-api.com/path/")

    def test_init_with_url_path_prefix(self):
        """Test that urljoin preserves the last path segment when URL has a custom prefix."""
        import urllib.parse

        # Simulate the issue: URL with path prefix (like /member1/deepseek_v4)
        kwargs = self.default_kwargs.copy()
        kwargs["url"] = "http://10.5.0.1:8080/member1/deepseek_v4"
        model = self.model_class(**kwargs)
        # urljoin should append endpoint to the full path, not drop deepseek_v4
        full_url = urllib.parse.urljoin(model.base_url, "v1/chat/completions")
        self.assertEqual(
            full_url,
            "http://10.5.0.1:8080/member1/deepseek_v4/v1/chat/completions",
        )

    def test_init_with_url_query_string(self):
        """Test that query strings are preserved when appending trailing slash."""
        kwargs = self.default_kwargs.copy()
        kwargs["url"] = "http://host.com/path?x=1"
        model = self.model_class(**kwargs)
        self.assertEqual(model.base_url, "http://host.com/path/?x=1")

    def test_init_with_url_fragment(self):
        """Test that fragments are preserved when appending trailing slash."""
        kwargs = self.default_kwargs.copy()
        kwargs["url"] = "http://host.com/path#section"
        model = self.model_class(**kwargs)
        self.assertEqual(model.base_url, "http://host.com/path/#section")

    def test_init_with_url_whitespace(self):
        """Test that leading/trailing whitespace is stripped from URL."""
        kwargs = self.default_kwargs.copy()
        kwargs["url"] = "  https://test-api.com/v1  "
        model = self.model_class(**kwargs)
        self.assertEqual(model.base_url, "https://test-api.com/v1/")

        # Empty/whitespace-only URL should fall through to host_ip:host_port
        kwargs["url"] = "   "
        model = self.model_class(**kwargs)
        self.assertEqual(model.base_url, "http://127.0.0.1:8000/")

    def test_init_with_ssl(self):
        kwargs = self.default_kwargs.copy()
        kwargs["enable_ssl"] = True
        model = self.model_class(**kwargs)
        self.assertEqual(model.base_url, "https://127.0.0.1:8000/")

    def test_get_base_url(self):
        model = self.model_class(**self.default_kwargs)
        self.assertEqual(model._get_base_url(), "http://127.0.0.1:8000/")

    def test_get_base_url_with_ipv6_host_ip(self):
        model = self.model_class(**self.ipv6_kwargs)
        self.assertEqual(model._get_base_url(), "http://[::1]:8000/")

    def test_get_base_url_with_hostname(self):
        kwargs = self.default_kwargs.copy()
        kwargs["host_ip"] = "localhost"
        model = self.model_class(**kwargs)
        self.assertEqual(model._get_base_url(), "http://localhost:8000/")

    @mock.patch("requests.get")
    def test_get_service_model_path_success(self, mock_get):
        mock_response = MockResponse(
            200, json_data={"data": [{"id": "test-model-123"}]}
        )
        mock_get.return_value = mock_response

        model = self.model_class(**self.default_kwargs)
        result = model._get_service_model_path()

        self.assertEqual(result, "test-model-123")
        mock_get.assert_called_once_with(
            "http://127.0.0.1:8000/v1/models",
            headers={"Content-Type": "application/json"},
            timeout=5,
        )

    @mock.patch("requests.get")
    def test_get_service_model_path_failure(self, mock_get):
        # 模拟requests.exceptions.RequestException异常
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        model = self.model_class(**self.default_kwargs)
        with self.assertRaises(AISBenchRuntimeError):
            model._get_service_model_path()

    def test_iter_lines_normal(self):
        model = self.model_class(**self.default_kwargs)
        stream = [b"line1\n\nline2\n\n", b"line3"]

        async def mock_stream():
            for chunk in stream:
                yield chunk

        async def run():
            lines = [line async for line in model.iter_lines(mock_stream())]
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], b"line1")
            self.assertEqual(lines[1], b"line2")
            self.assertEqual(lines[2], b"line3")

        asyncio.run(run())

    def test_iter_lines_crlf(self):
        model = self.model_class(**self.default_kwargs)
        stream = [b"line1\r\n\r\nline2\r\rline3"]

        async def mock_stream():
            for chunk in stream:
                yield chunk

        async def run():
            lines = [line async for line in model.iter_lines(mock_stream())]
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], b"line1")
            self.assertEqual(lines[1], b"line2")
            self.assertEqual(lines[2], b"line3")

        asyncio.run(run())

    @mock.patch("aiohttp.ClientSession.post")
    async def test_text_infer_success(self, mock_post):
        # 设置mock响应
        mock_response = mock.AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = json.dumps({"response": "test output"})
        mock_post.return_value.__aenter__.return_value = mock_response

        model = self.model_class(**self.default_kwargs)
        model.session = mock.AsyncMock()
        output = Output()

        await model.text_infer({"prompt": "test"}, output)

        self.assertTrue(output.success)
        self.assertEqual(output.text, "test output")

    @mock.patch("aiohttp.ClientSession.post")
    async def test_text_infer_failure(self, mock_post):
        mock_response = mock.AsyncMock()
        mock_response.status = 400
        mock_response.reason = "Bad Request"
        mock_post.return_value.__aenter__.return_value = mock_response

        model = self.model_class(**self.default_kwargs)
        model.session = mock.AsyncMock()
        output = Output()

        await model.text_infer({"prompt": "test"}, output)

        self.assertFalse(output.success)
        self.assertEqual(output.error_info, "Bad Request")

    @mock.patch("aiohttp.ClientSession.post")
    async def test_text_infer_json_decode_error(self, mock_post):
        mock_response = mock.AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = "invalid json"
        mock_post.return_value.__aenter__.return_value = mock_response

        model = self.model_class(**self.default_kwargs)
        model.session = mock.AsyncMock()
        output = Output()

        with self.assertRaises(AISBenchValueError):
            await model.text_infer({"prompt": "test"}, output)

    @mock.patch("aiohttp.ClientSession.post")
    async def test_stream_infer_success(self, mock_post):
        # 创建模拟的流响应
        async def mock_content():
            yield b'data: {"chunk": "hello"}\n\n'
            yield b'data: {"chunk": " world"}\n\n'
            yield b"data: [DONE]\n\n"

        mock_response = mock.AsyncMock()
        mock_response.status = 200
        mock_response.content.__aiter__.return_value = mock_content()
        mock_post.return_value.__aenter__.return_value = mock_response

        model = self.model_class(**self.default_kwargs)
        model.stream = True
        model.session = mock.AsyncMock()
        output = Output()

        async def run():
            await model.stream_infer({"prompt": "test"}, output)
            self.assertTrue(output.success)
            self.assertEqual(output.text, "hello world")

        asyncio.run(run())

    @mock.patch("aiohttp.ClientSession.post")
    async def test_stream_infer_json_decode_error(self, mock_post):
        async def mock_content():
            yield b"data: invalid json\n\n"

        mock_response = mock.AsyncMock()
        mock_response.status = 200
        mock_response.content.__aiter__.return_value = mock_content()
        mock_post.return_value.__aenter__.return_value = mock_response

        model = self.model_class(**self.default_kwargs)
        model.stream = True
        model.session = mock.AsyncMock()
        output = Output()

        async def run():
            with self.assertRaises(AISBenchValueError):
                await model.stream_infer({"prompt": "test"}, output)

        asyncio.run(run())

    @mock.patch("aiohttp.ClientSession")
    def test_generate_with_session(self, mock_session_class):
        mock_session = mock.AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session

        model = self.model_class(**self.default_kwargs)
        output = Output()

        async def run():
            # 使用mock模拟generate方法的关键部分
            with mock.patch.object(model, "text_infer") as mock_text_infer:
                await model.generate("test prompt", 100, output, session=mock_session)
                mock_text_infer.assert_called_once()

        asyncio.run(run())

    @mock.patch("aiohttp.ClientSession")
    def test_generate_with_retry(self, mock_session_class):

        mock_session = mock.AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session

        model = self.model_class(**self.default_kwargs)
        model.retry = 2
        output = Output()

        # 第一次调用失败，第二次成功
        mock_text_infer = mock.AsyncMock(side_effect=[Exception("First fail"), None])

        async def run():
            with mock.patch.object(model, "text_infer", mock_text_infer):
                await model.generate("test prompt", 100, output, session=mock_session)
                self.assertEqual(mock_text_infer.call_count, 2)

        asyncio.run(run())

    def test_abstract_methods(self):
        # 测试抽象方法是否正确抛出异常
        model = self.model_class(**self.default_kwargs)

        # 这些方法在子类中已经实现，所以不会抛出异常
        # 这里主要是测试基类的行为
        pass


class TestPrefillConcurrency(unittest.TestCase):
    def setUp(self):
        class ConcreteAPIModel(BaseAPIModel):
            def _get_url(self):
                return self.base_url + "test"

            async def get_request_body(self, input_data, max_out_len, output, **args):
                return {"prompt": input_data, "max_tokens": max_out_len}

            async def parse_stream_response(self, data, output):
                output.text += data.get("chunk", "")

        self.model_class = ConcreteAPIModel

    def _make_model(self, retry=1):
        model = self.model_class(
            path="test-model",
            stream=True,
            retry=retry,
            host_ip="127.0.0.1",
            host_port=8000,
        )
        model.session = mock.MagicMock()
        return model

    def _assert_permit_available(self, shared_limiter):
        self.assertEqual(shared_limiter.inflight.value, 0)
        self.assertEqual(shared_limiter.waiting.value, 0)
        self.assertTrue(shared_limiter.semaphore.acquire(timeout=0))
        shared_limiter.semaphore.release()

    def test_lease_release_is_idempotent(self):
        async def run():
            shared_limiter = SharedPrefillLimiter(1)
            limiter = AsyncPrefillLimiter(shared_limiter)
            lease = PrefillLease(shared_limiter)
            try:
                await limiter.acquire(lease)
                self.assertTrue(lease.held)
                self.assertEqual(shared_limiter.inflight.value, 1)
                lease.release()
                lease.release()
                self._assert_permit_available(shared_limiter)
                self.assertEqual(shared_limiter.max_inflight.value, 1)
            finally:
                limiter.close()

        asyncio.run(run())

    def test_cancelled_waiter_returns_late_acquisition(self):
        class UncancellableFuture(Future):
            def cancel(self):
                return False

        async def run():
            shared_limiter = SharedPrefillLimiter(1)
            limiter = AsyncPrefillLimiter(shared_limiter)
            lease = PrefillLease(shared_limiter)
            job = UncancellableFuture()
            try:
                with mock.patch.object(limiter.executor, "submit", return_value=job):
                    task = asyncio.create_task(limiter.acquire(lease))
                    await asyncio.sleep(0)
                    self.assertEqual(shared_limiter.waiting.value, 1)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

                # Simulate the uncancellable thread acquiring after coroutine cancellation.
                self.assertTrue(shared_limiter.semaphore.acquire(timeout=0))
                job.set_result(True)
                await asyncio.sleep(0)
                self.assertFalse(lease.held)
                self._assert_permit_available(shared_limiter)
            finally:
                limiter.close()

        asyncio.run(run())

    def test_stream_errors_and_empty_stream_release_permit(self):
        async def run():
            shared_limiter = SharedPrefillLimiter(1)
            limiter = AsyncPrefillLimiter(shared_limiter)
            model = self._make_model()
            model.set_prefill_limiter(limiter)
            cases = [
                ("http_error", _make_stream_context(status=503, reason="Unavailable"), None, False),
                ("empty_stream", _make_stream_context(), None, True),
                (
                    "invalid_json",
                    _make_stream_context([b"data: invalid json\n\n"]),
                    AISBenchValueError,
                    False,
                ),
                ("connection_error", RuntimeError("connection failed"), RuntimeError, False),
            ]
            try:
                for name, post_result, expected_error, expected_success in cases:
                    with self.subTest(name=name):
                        model.session.post.reset_mock()
                        if isinstance(post_result, Exception):
                            model.session.post.side_effect = post_result
                        else:
                            model.session.post.side_effect = None
                            model.session.post.return_value = post_result
                        output = Output()
                        if expected_error is None:
                            await model.stream_infer({}, output)
                        else:
                            with self.assertRaises(expected_error):
                                await model.stream_infer({}, output)
                        self.assertEqual(output.success, expected_success)
                        self._assert_permit_available(shared_limiter)
                        self.assertEqual(len(output.extra_perf_data["prefill_wait_time"]), 1)
            finally:
                limiter.close()

        asyncio.run(run())

    def test_parse_error_after_first_token_does_not_over_release(self):
        async def run():
            shared_limiter = SharedPrefillLimiter(1)
            limiter = AsyncPrefillLimiter(shared_limiter)
            model = self._make_model()
            model.set_prefill_limiter(limiter)
            model.session.post.return_value = _make_stream_context(
                [b'data: {"chunk": "first"}\n\n']
            )
            model.parse_stream_response = mock.AsyncMock(
                side_effect=RuntimeError("parse failed")
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "parse failed"):
                    await model.stream_infer({}, Output())
                self._assert_permit_available(shared_limiter)
            finally:
                limiter.close()

        asyncio.run(run())

    def test_cancellation_before_first_token_releases_permit(self):
        async def run():
            shared_limiter = SharedPrefillLimiter(1)
            limiter = AsyncPrefillLimiter(shared_limiter)
            model = self._make_model()
            model.set_prefill_limiter(limiter)
            response_started = asyncio.Event()
            context = _make_stream_context()
            response = context.__aenter__.return_value

            async def content():
                response_started.set()
                await asyncio.Event().wait()
                yield b""

            response.content = content()
            model.session.post.return_value = context
            request_task = asyncio.create_task(model.stream_infer({}, Output()))
            try:
                await asyncio.wait_for(response_started.wait(), timeout=0.5)
                self.assertEqual(shared_limiter.inflight.value, 1)
                request_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await request_task
                self._assert_permit_available(shared_limiter)
            finally:
                if not request_task.done():
                    request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)
                limiter.close()

        asyncio.run(run())

    def test_permit_is_released_before_decode_finishes(self):
        async def run():
            shared_limiter = SharedPrefillLimiter(1)
            limiter = AsyncPrefillLimiter(shared_limiter)
            model = self._make_model()
            model.set_prefill_limiter(limiter)
            decode_continue = asyncio.Event()
            context = _make_stream_context()
            response = context.__aenter__.return_value

            async def content():
                yield b'data: {"chunk": "first"}\n\n'
                await decode_continue.wait()
                yield b"data: [DONE]\n\n"

            response.content = content()
            model.session.post.return_value = context
            output = Output()
            request_task = asyncio.create_task(model.stream_infer({}, output))
            next_lease = PrefillLease(shared_limiter)
            try:
                for _ in range(100):
                    if shared_limiter.max_inflight.value == 1 and shared_limiter.inflight.value == 0:
                        break
                    await asyncio.sleep(0.01)
                else:
                    self.fail("First-token permit was not released")

                await asyncio.wait_for(limiter.acquire(next_lease), timeout=0.5)
                self.assertFalse(request_task.done())
                next_lease.release()
                decode_continue.set()
                await asyncio.wait_for(request_task, timeout=0.5)
                self.assertEqual(output.text, "first")
                self._assert_permit_available(shared_limiter)
            finally:
                decode_continue.set()
                next_lease.release()
                if not request_task.done():
                    request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)
                limiter.close()

        asyncio.run(run())

    def test_retry_acquires_a_new_lease_for_each_attempt(self):
        async def run():
            shared_limiter = SharedPrefillLimiter(1)
            limiter = AsyncPrefillLimiter(shared_limiter)
            model = self._make_model(retry=2)
            model.set_prefill_limiter(limiter)
            model.session.post.side_effect = [
                RuntimeError("first attempt failed"),
                _make_stream_context([
                    b'data: {"chunk": "success"}\n\n',
                    b"data: [DONE]\n\n",
                ]),
            ]
            output = Output()
            try:
                await model.generate("test", 100, output, session=model.session)
                self.assertTrue(output.success)
                self.assertEqual(output.text, "success")
                self.assertEqual(model.session.post.call_count, 2)
                self.assertEqual(len(output.extra_perf_data["prefill_wait_time"]), 2)
                self._assert_permit_available(shared_limiter)
            finally:
                limiter.close()

        asyncio.run(run())

    def test_multiprocess_workers_share_one_global_limit(self):
        ctx = mp.get_context()
        shared_limiter = SharedPrefillLimiter(2, ctx=ctx)
        start_event = ctx.Event()
        processes = [
            ctx.Process(
                target=_prefill_process_worker,
                args=(shared_limiter, start_event, 20),
            )
            for _ in range(4)
        ]
        try:
            for process in processes:
                process.start()
            start_event.set()
            for process in processes:
                process.join(timeout=30)
            for process in processes:
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(shared_limiter.max_inflight.value, 2)
            self._assert_permit_available(shared_limiter)
        finally:
            start_event.set()
            for process in processes:
                if process.pid is None:
                    continue
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)


class TestAPITemplateParser(unittest.TestCase):
    def setUp(self):
        self.default_meta_template = {
            "round": [
                {"role": "user", "api_role": "user"},
                {"role": "assistant", "api_role": "assistant", "generate": True},
            ]
        }

    def test_init_with_valid_meta_template(self):
        parser = APITemplateParser(self.default_meta_template)
        self.assertEqual(len(parser.roles), 2)
        self.assertIn("user", parser.roles)
        self.assertIn("assistant", parser.roles)

    def test_init_with_invalid_meta_template(self):
        # 测试缺少round字段
        invalid_template = {}
        # 只有当meta_template不为None时才会检查round字段
        parser = APITemplateParser(invalid_template)

        # 测试round字段类型错误
        invalid_template = {"round": "not a list"}
        with self.assertRaises(AISBenchTypeError):
            APITemplateParser(invalid_template)

    def test_init_with_reserved_roles(self):
        meta_template = deepcopy(self.default_meta_template)
        meta_template["reserved_roles"] = [{"role": "system", "api_role": "system"}]
        parser = APITemplateParser(meta_template)
        self.assertEqual(len(parser.roles), 3)
        self.assertIn("system", parser.roles)

    def test_init_with_duplicate_roles(self):
        meta_template = {
            "round": [
                {"role": "user", "api_role": "user"},
                {"role": "user", "api_role": "user"},  # 重复的角色
            ]
        }
        with self.assertRaises(AISBenchTypeError):
            APITemplateParser(meta_template)

    def test_parse_template_string(self):
        parser = APITemplateParser(self.default_meta_template)
        result = parser.parse_template("test prompt", "gen")
        self.assertEqual(result, "test prompt")

    def test_parse_template_invalid_type(self):
        parser = APITemplateParser(self.default_meta_template)
        with self.assertRaises(AISBenchTypeError):
            parser.parse_template(123, "gen")  # 类型错误

    def test_parse_template_invalid_mode(self):
        parser = APITemplateParser(self.default_meta_template)
        with self.assertRaises(AISBenchTypeError):
            parser.parse_template("test", "invalid_mode")

    def test_parse_template_without_meta_template(self):
        parser = APITemplateParser(None)
        # 使用PromptList而不是普通列表
        prompt_list = PromptList([{"prompt": "Hello"}, {"prompt": "World"}])
        result = parser.parse_template(prompt_list, "gen")
        self.assertEqual(result, "Hello\nWorld")

    def test_update_role_dict(self):
        parser = APITemplateParser(self.default_meta_template)
        prompts = [{"role": "user", "prompt": "test"}]
        role_dict = parser._update_role_dict(prompts)
        self.assertEqual(role_dict["user"]["prompt"], "test")

    def test_update_role_dict_with_fallback(self):
        parser = APITemplateParser(self.default_meta_template)
        prompts = [{"role": "unknown", "fallback_role": "user", "prompt": "test"}]
        role_dict = parser._update_role_dict(prompts)
        self.assertEqual(role_dict["user"]["prompt"], "test")

    def test_split_rounds(self):
        parser = APITemplateParser(self.default_meta_template)
        prompt_template = [
            {"role": "user", "prompt": "q1"},
            {"role": "assistant", "prompt": "a1"},
            {"role": "user", "prompt": "q2"},
            {"role": "assistant", "prompt": "a2"},
        ]
        result = parser._split_rounds(
            prompt_template, self.default_meta_template["round"]
        )
        self.assertEqual(result, [0, 2, 4])

    def test_split_rounds_with_fallback_role(self):
        parser = APITemplateParser(self.default_meta_template)
        prompt_template = [
            {"role": "unknown", "fallback_role": "user", "prompt": "q1"},
            {"role": "assistant", "prompt": "a1"},
        ]
        result = parser._split_rounds(
            prompt_template, self.default_meta_template["round"]
        )
        self.assertEqual(result, [0, 2])

    def test_split_rounds_invalid_role(self):
        parser = APITemplateParser(self.default_meta_template)
        prompt_template = [
            {"role": "unknown", "prompt": "q1"}  # 没有fallback_role
        ]
        with self.assertRaises(AISBenchKeyError):
            parser._split_rounds(prompt_template, self.default_meta_template["round"])

    def test_prompt2api_string(self):
        parser = APITemplateParser(self.default_meta_template)
        result, cont = parser._prompt2api("test", {}, False)
        self.assertEqual(result, "test")
        self.assertTrue(cont)

    def test_prompt2api_dict(self):
        parser = APITemplateParser(self.default_meta_template)
        parser.roles["user"] = {"role": "user", "api_role": "user", "prompt": "test"}
        result, cont = parser._prompt2api({"role": "user"}, parser.roles, False)
        self.assertEqual(result["role"], "user")
        self.assertTrue(cont)

    def test_prompt2api_list_with_string(self):
        parser = APITemplateParser(self.default_meta_template)
        with self.assertRaises(AISBenchTypeError):
            parser._prompt2api(["test"], {}, False)

    def test_role2api_role(self):
        parser = APITemplateParser(self.default_meta_template)
        parser.roles["user"] = {
            "role": "user",
            "api_role": "user",
            "prompt": "test",
            "begin": "<s>",
            "end": "</s>",
        }
        result, cont = parser._role2api_role({"role": "user"}, parser.roles, False)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["prompt"], "<s>test</s>")
        self.assertTrue(cont)

    def test_role2api_role_for_gen(self):
        parser = APITemplateParser(self.default_meta_template)
        parser.roles["assistant"] = {
            "role": "assistant",
            "api_role": "assistant",
            "generate": True,
        }
        result, cont = parser._role2api_role({"role": "assistant"}, parser.roles, True)
        self.assertIsNone(result)
        self.assertFalse(cont)

    def test_role2api_role_invalid_content(self):
        parser = APITemplateParser(self.default_meta_template)
        parser.roles["user"] = {
            "role": "user",
            "api_role": "user",
        }  # 缺少prompt或prompt_mm
        with self.assertRaises(AISBenchValueError):
            parser._role2api_role({"role": "user"}, parser.roles, False)

    def test_role2api_role_with_prompt_mm(self):
        parser = APITemplateParser(self.default_meta_template)
        parser.roles["user"] = {
            "role": "user",
            "api_role": "user",
            "prompt_mm": [{"type": "text", "content": "test"}],
        }
        result, cont = parser._role2api_role({"role": "user"}, parser.roles, False)
        self.assertEqual(result["prompt"], [{"type": "text", "content": "test"}])

    def test_parse_template_with_meta_template(self):
        meta_template = {
            "begin": {"role": "system", "prompt": "System prompt"},
            "round": [
                {"role": "user", "api_role": "user"},
                {"role": "assistant", "api_role": "assistant"},
            ],
        }
        parser = APITemplateParser(meta_template)

        # 模拟一个完整的对话结构
        prompt_template = PromptList(
            [
                {"section": "round", "pos": "begin"},
                {"role": "user", "prompt": "Question 1"},
                {"role": "assistant", "prompt": "Answer 1"},
                {"section": "round", "pos": "end"},
            ]
        )

        result = parser.parse_template(prompt_template, "gen")
        self.assertEqual(len(result), 3)  # system + user + assistant
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[2]["role"], "assistant")


class TestBaseAPIModelParseLogprobs(unittest.TestCase):
    def test_parse_logprobs_base_default_no_op(self):
        """测试 BaseAPIModel._parse_logprobs 基类空实现不报错且不修改 output"""
        import asyncio
        from unittest.mock import MagicMock

        # BaseAPIModel 需要初始化参数，用 mock 绕过
        model = MagicMock(spec=BaseAPIModel)
        # 绑定真实方法到 mock 对象上
        model._parse_logprobs = BaseAPIModel._parse_logprobs.__get__(model, BaseAPIModel)

        output = Output()
        output.origin_logprobs = []

        choice = {"logprobs": {"tokens": ["A"], "token_logprobs": [-0.5]}}

        asyncio.run(model._parse_logprobs(choice, output))

        self.assertEqual(output.origin_logprobs, [])


if __name__ == "__main__":
    unittest.main()
