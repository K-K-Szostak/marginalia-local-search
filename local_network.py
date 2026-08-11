from __future__ import annotations

import ipaddress
import os
import urllib.parse


def _is_loopback(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def loopback_host(environment_name: str, default: str = "127.0.0.1") -> str:
    value = os.getenv(environment_name, default).strip()
    hostname = value[1:-1] if value.startswith("[") and value.endswith("]") else value
    if not _is_loopback(hostname):
        raise RuntimeError(f"{environment_name} must identify this computer's loopback interface")
    return value


def loopback_url(environment_name: str, default: str) -> str:
    value = os.getenv(environment_name, default).strip()
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or not _is_loopback(parsed.hostname)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            f"{environment_name} must be a plain http://localhost or loopback-IP URL without credentials or a path"
        )
    try:
        parsed.port
    except ValueError as exc:
        raise RuntimeError(f"{environment_name} contains an invalid port") from exc
    return value.rstrip("/")
