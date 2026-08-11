from __future__ import annotations

import http.client
import os
import threading
import unittest
from unittest.mock import patch

import app.server as server
from local_network import loopback_host, loopback_url


class LoopbackConfigurationTests(unittest.TestCase):
    def test_loopback_hosts_are_accepted(self):
        for value in ("localhost", "127.0.0.1", "127.9.8.7", "::1", "[::1]"):
            with self.subTest(value=value), patch.dict(os.environ, {"TEST_HOST": value}):
                self.assertEqual(loopback_host("TEST_HOST"), value)

    def test_non_loopback_hosts_are_rejected(self):
        for value in ("0.0.0.0", "192.168.1.20", "example.com"):
            with self.subTest(value=value), patch.dict(os.environ, {"TEST_HOST": value}):
                with self.assertRaises(RuntimeError):
                    loopback_host("TEST_HOST")

    def test_only_plain_loopback_ollama_urls_are_accepted(self):
        accepted=("http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434/")
        for value in accepted:
            with self.subTest(value=value), patch.dict(os.environ, {"TEST_URL": value}):
                self.assertEqual(loopback_url("TEST_URL", "http://127.0.0.1"), value.rstrip("/"))
        rejected=(
            "http://0.0.0.0:11434", "http://192.168.1.20:11434", "http://ollama.example:11434",
            "https://127.0.0.1:11434", "http://user:pass@127.0.0.1:11434", "http://127.0.0.1:11434/api",
        )
        for value in rejected:
            with self.subTest(value=value), patch.dict(os.environ, {"TEST_URL": value}):
                with self.assertRaises(RuntimeError):
                    loopback_url("TEST_URL", "http://127.0.0.1")


class LocalApiSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_token=server.SESSION_TOKEN
        server.SESSION_TOKEN="test-session-token"
        cls.httpd=server.SingleInstanceServer(("127.0.0.1",0),server.Handler)
        cls.port=cls.httpd.server_port
        cls.worker=threading.Thread(target=cls.httpd.serve_forever,daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown(); cls.httpd.server_close(); cls.worker.join(timeout=5)
        server.SESSION_TOKEN=cls.original_token

    def request(self, method, path, *, headers=None, body=None):
        connection=http.client.HTTPConnection("127.0.0.1",self.port,timeout=5)
        connection.request(method,path,body=body,headers=headers or {})
        response=connection.getresponse()
        payload=response.read()
        result=(response.status,dict(response.getheaders()),payload)
        connection.close()
        return result

    def authorized_headers(self):
        return {"Host":f"127.0.0.1:{self.port}","X-Marginalia-Token":"test-session-token"}

    def test_valid_token_allows_api_request(self):
        status,headers,_=self.request("GET","/api/app-instance",headers=self.authorized_headers())
        self.assertEqual(status,200)
        self.assertIn("frame-ancestors 'none'",headers["Content-Security-Policy"])
        self.assertEqual(headers["X-Frame-Options"],"DENY")

    def test_api_rejects_missing_token(self):
        status,_,_=self.request("GET","/api/app-instance",headers={"Host":f"127.0.0.1:{self.port}"})
        self.assertEqual(status,403)

    def test_api_accepts_same_site_session_cookie(self):
        headers={"Host":f"localhost:{self.port}","Cookie":f"marginalia_session_{self.port}=test-session-token"}
        status,_,_=self.request("GET","/api/app-instance",headers=headers)
        self.assertEqual(status,200)

    def test_host_header_attack_is_rejected(self):
        headers=self.authorized_headers(); headers["Host"]="attacker.example"
        status,_,_=self.request("GET","/api/app-instance",headers=headers)
        self.assertEqual(status,421)

    def test_cross_origin_request_is_rejected(self):
        headers=self.authorized_headers(); headers["Origin"]="https://attacker.example"
        status,_,_=self.request("GET","/api/app-instance",headers=headers)
        self.assertEqual(status,403)

    def test_simple_cross_origin_content_type_is_rejected(self):
        headers=self.authorized_headers(); headers["Content-Type"]="text/plain"
        status,_,_=self.request("POST","/api/ai/skip",headers=headers,body=b"{}")
        self.assertEqual(status,415)


if __name__ == "__main__":
    unittest.main()
