"""Real HTTP test client, using only app dependencies and the standard library.

Avoid coupling API tests to Starlette's optional httpx/httpx2 test-client version.
It also covers sockets, session cookies, middleware and application lifespan.
"""
import http.cookiejar
import json as json_module
import socket
import threading
import time
import urllib.error
import urllib.request

import uvicorn


class Response:
    def __init__(self, response):
        self.status_code = response.code
        self.headers = response.headers
        self.content = response.read()

    def json(self):
        return json_module.loads(self.content)


class LocalClient:
    def __init__(self, app):
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        self.base_url = f'http://127.0.0.1:{sock.getsockname()[1]}'
        self.server = uvicorn.Server(uvicorn.Config(app, log_level='error', access_log=False, lifespan='on'))
        self.thread = threading.Thread(target=self.server.run, kwargs={'sockets': [sock]}, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                self.close()
                raise RuntimeError('Test backend failed to start')
            time.sleep(0.01)
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
        )

    def request(self, method, path, *, json=None, headers=None):
        data = None if json is None else json_module.dumps(json).encode('utf-8')
        headers = dict(headers or {})
        if data is not None:
            headers.setdefault('Content-Type', 'application/json')
        request = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        try:
            with self.opener.open(request, timeout=10) as response:
                return Response(response)
        except urllib.error.HTTPError as response:
            with response:
                return Response(response)

    def get(self, path, **kwargs):
        return self.request('GET', path, **kwargs)

    def post(self, path, **kwargs):
        return self.request('POST', path, **kwargs)

    def put(self, path, **kwargs):
        return self.request('PUT', path, **kwargs)

    def close(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('Test backend failed to shut down')
