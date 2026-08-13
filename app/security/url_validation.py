"""
Healix - URL validation for outbound fetches (SSRF mitigation).

Used by speech-to-text when downloading remote audio. Blocks private/reserved
address ranges and optional host allowlists.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Iterable, Optional, Set
from urllib.parse import urlparse

from app.exceptions import AudioDownloadError

_BLOCKED_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)

_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata.google",
    }
)


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
        return True
    return any(ip in network for network in _BLOCKED_NETWORKS)


def _normalize_allowed_hosts(hosts: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if not hosts:
        return None
    normalized = {host.strip().lower() for host in hosts if host and host.strip()}
    return normalized or None


def validate_outbound_http_url(
    url: str,
    *,
    allowed_hosts: Optional[Iterable[str]] = None,
    purpose: str = "download",
) -> None:
    """Raise ``AudioDownloadError`` when *url* is unsafe for server-side fetch."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise AudioDownloadError(
            f"Only http/https URLs are allowed for audio {purpose}."
        )

    hostname = parsed.hostname
    if not hostname:
        raise AudioDownloadError("Audio URL must include a hostname.")

    host_lower = hostname.lower().rstrip(".")
    if host_lower in _BLOCKED_HOSTNAMES:
        raise AudioDownloadError("Audio URL hostname is not allowed.")

    allowlist = _normalize_allowed_hosts(allowed_hosts)
    if allowlist is not None and host_lower not in allowlist:
        raise AudioDownloadError("Audio URL hostname is not in the allowlist.")

    # Literal IP in URL — validate before DNS to avoid bypass via numeric host.
    try:
        literal_ip = ipaddress.ip_address(host_lower.strip("[]"))
    except ValueError:
        literal_ip = None
    else:
        if _is_blocked_ip(literal_ip):
            raise AudioDownloadError("Audio URL resolves to a blocked address.")

    try:
        addr_infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise AudioDownloadError("Audio URL hostname could not be resolved.") from exc

    resolved_ips = {
        info[4][0]
        for info in addr_infos
        if info and info[4]
    }
    if not resolved_ips:
        raise AudioDownloadError("Audio URL hostname could not be resolved.")

    for resolved in resolved_ips:
        try:
            ip = ipaddress.ip_address(resolved)
        except ValueError as exc:
            raise AudioDownloadError("Audio URL resolved to an invalid address.") from exc
        if _is_blocked_ip(ip):
            raise AudioDownloadError("Audio URL resolves to a blocked address.")
