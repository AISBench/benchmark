"""Direct HTTP probes with bounded bodies, no redirects and a total deadline."""
import hashlib
import http.client
import json
import socket
import threading
import time
from urllib.parse import urlsplit
from .json_io import unique_object, reject_constant
from .process_guard import remaining, Cancelled
from .protocol import ProtocolError, now


def request(url, deadline, cancelled, *, payload=None, max_bytes=65536):
    parts = urlsplit(url)
    if (parts.scheme not in ('http','https') or not parts.hostname or parts.username or
            parts.password or parts.fragment or any(ord(c)<33 for c in url) or
            not 1 <= max_bytes <= 65536):
        raise ProtocolError('INVALID_INPUT: HTTP probe target')
    if cancelled.is_set():
        raise Cancelled('HTTP probe cancelled')
    timeout = remaining(deadline)
    connection = (http.client.HTTPSConnection if parts.scheme=='https' else http.client.HTTPConnection)(
        parts.hostname,parts.port,timeout=timeout)
    done, stopped = threading.Event(), threading.Event()
    result = []
    body = None if payload is None else json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8')
    if body is not None and len(body)>65536:
        raise ProtocolError('INVALID_INPUT: HTTP request exceeds budget')
    def check():
        remaining(deadline)
        if cancelled.is_set() or stopped.is_set():
            raise Cancelled('HTTP probe cancelled')
    def work():
        response = None
        try:
            connection.connect()
            check()  # DNS/TLS may outlive a caller deadline; never send afterwards.
            connection.request('GET' if body is None else 'POST',(parts.path or '/') + ('?'+parts.query if parts.query else ''),
                               body=body,headers={'Content-Type':'application/json','Connection':'close'})
            response = connection.getresponse()
            data = bytearray()
            while True:
                check()
                block = response.read1(min(8192,max_bytes+1-len(data)))
                if not block:
                    break
                data.extend(block)
                if len(data)>max_bytes:
                    raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: HTTP body exceeds budget')
            check()
            result.append((response.status,bytes(data)))
        except Exception as error:
            result.append(error)
        finally:
            if response is not None:
                response.close()
            connection.close()
            done.set()
    worker = threading.Thread(target=work,daemon=True)
    worker.start()
    try:
        while not done.wait(min(.05,remaining(deadline))):
            check()
        check()
        if isinstance(result[0],Exception):
            raise result[0]
        return result[0]
    finally:
        stopped.set()
        sock = connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()


def wait_health(urls, deadline, cancelled):
    evidence = []
    for url in dict.fromkeys(urls):
        while True:
            remaining(deadline)
            try:
                status, _ = request(url,min(deadline,time.monotonic()+5),cancelled,max_bytes=8192)
                if status==200:
                    evidence.append(dict(url=url,status=200,time=now()))
                    break
                if 300 <= status < 400:
                    raise ProtocolError('INVALID_INPUT: health redirects are not followed')
            except (OSError,TimeoutError,http.client.HTTPException):
                remaining(deadline)
            if cancelled.wait(min(.2,remaining(deadline))):
                raise Cancelled('Health check cancelled')
    return evidence


def smoke_completion(endpoint, model, deadline, cancelled):
    payload = dict(model=model,prompt='Hi',max_tokens=16,temperature=0,stream=False)
    status, raw = request(endpoint.rstrip('/')+'/v1/completions',deadline,cancelled,payload=payload)
    if status != 200:
        raise ProtocolError('INVALID_INPUT: smoke HTTP response unsuccessful')
    try:
        value = json.loads(raw,object_pairs_hook=unique_object,parse_constant=reject_constant)
        choices = value['choices']
        if not isinstance(choices,list) or not choices or not isinstance(choices[0]['text'],str) or not choices[0]['text']:
            raise ValueError('missing completion')
    except (ValueError,KeyError,TypeError) as exc:
        raise ProtocolError('INVALID_INPUT: smoke completion evidence invalid') from exc
    return dict(status='VERIFIED',time=now(),http_status=status,model=model,
                request=payload,response_sha256=hashlib.sha256(raw).hexdigest(),response_bytes=len(raw))
