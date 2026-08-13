"""Unit tests for SSRF URL validation used by speech-to-text."""

import pytest

from app.exceptions import AudioDownloadError
from app.security.url_validation import validate_outbound_http_url


def test_blocks_localhost_hostname():
    with pytest.raises(AudioDownloadError, match="not allowed"):
        validate_outbound_http_url("http://localhost/audio.wav")


def test_blocks_private_literal_ip():
    with pytest.raises(AudioDownloadError, match="blocked"):
        validate_outbound_http_url("http://127.0.0.1/audio.wav")


def test_blocks_metadata_hostname():
    with pytest.raises(AudioDownloadError, match="not allowed"):
        validate_outbound_http_url("http://metadata.google.internal/computeMetadata/v1/")


def test_rejects_non_http_scheme():
    with pytest.raises(AudioDownloadError, match="http/https"):
        validate_outbound_http_url("file:///etc/passwd")


def test_enforces_host_allowlist(monkeypatch):
    def fake_getaddrinfo(hostname, port, family=0, type=0, proto=0, flags=0):
        assert hostname == "cdn.example.com"
        return [(2, 1, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr("app.security.url_validation.socket.getaddrinfo", fake_getaddrinfo)

    validate_outbound_http_url(
        "https://cdn.example.com/audio.wav",
        allowed_hosts={"cdn.example.com"},
    )

    with pytest.raises(AudioDownloadError, match="allowlist"):
        validate_outbound_http_url(
            "https://evil.example.com/audio.wav",
            allowed_hosts={"cdn.example.com"},
        )


def test_allows_public_host_after_dns_resolution(monkeypatch):
    def fake_getaddrinfo(hostname, port, family=0, type=0, proto=0, flags=0):
        return [(2, 1, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr("app.security.url_validation.socket.getaddrinfo", fake_getaddrinfo)

    validate_outbound_http_url("https://cdn.example.com/audio.wav")
