"""Protocol checks use exact bytes; SQL and store semantics live in integration tests."""

import hashlib
import hmac

import pytest

pytest.importorskip("sqlalchemy")

from github_intelligence.application.runtime import Runtime
from github_intelligence.application.webhooks import verify_signature
from github_intelligence.config import Settings


def test_exact_raw_bytes_and_constant_time_comparison(monkeypatch):
    raw = b'{"text":"caf\xc3\xa9"}\n'
    secret = "a" * 32
    signature = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    original = hmac.compare_digest
    calls = []

    def compare(left, right):
        calls.append((left, right))
        return original(left, right)

    monkeypatch.setattr(hmac, "compare_digest", compare)
    assert verify_signature(raw, signature, secret)
    assert not verify_signature(raw.rstrip(), signature, secret)
    assert not verify_signature(raw, None, secret)
    assert not verify_signature(raw, "sha256=\u00e9", secret)
    assert len(calls) == 4
    assert calls[0] == (signature.encode(), signature.encode())


@pytest.mark.parametrize("secret", [None, "", "too-short"])
def test_enabled_webhooks_reject_invalid_secret_without_exposing_it(secret):
    settings = Settings(_env_file=None, github_webhooks_enabled=True, github_webhook_secret=secret)
    with pytest.raises(ValueError, match="at least 32 bytes") as failure:
        Runtime(settings)
    if secret:
        assert secret not in str(failure.value)
        assert secret not in repr(settings)


def test_secret_is_hidden_in_settings_repr():
    secret = "never-display-this-secret" * 2
    assert secret not in repr(Settings(_env_file=None, github_webhook_secret=secret))
