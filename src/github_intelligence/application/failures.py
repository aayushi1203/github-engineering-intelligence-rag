"""Controlled categories; never persist raw exception text or provider payloads."""

from dataclasses import dataclass

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy.exc import InterfaceError, OperationalError

from github_intelligence.ingestion.service import IngestionError


@dataclass(frozen=True)
class Failure:
    code: str
    retryable: bool
    minimum_delay: float = 0


class ControlledFailure(RuntimeError):
    def __init__(self, code: str, retryable: bool):
        self.code, self.retryable = code, retryable
        super().__init__(code)


def classify(error: Exception) -> Failure:
    if isinstance(error, ControlledFailure):
        return Failure(error.code, error.retryable)
    if isinstance(error, IngestionError):
        details = error.report.failure_details
        status = details.get("status_code")
        transient = error.code in {
            "transport_failure",
            "primary_rate_limit",
            "secondary_rate_limit",
            "server_error",
            "transient_server_error",
        } or status in {429, 500, 502, 503, 504}
        if not transient and status in {401, 403}:
            return Failure("github_authorization_rejected", False)
        if status == 404:
            return Failure("github_repository_not_found", False)
        return Failure(
            "github_transient" if transient else "github_rejected",
            transient,
            float(details.get("retry_after") or 0),
        )
    if isinstance(error, (OperationalError, InterfaceError)):
        return Failure("database_unavailable", True)
    if isinstance(error, ResponseHandlingException):
        temporary = isinstance(error.source, (httpx.TransportError, OSError, TimeoutError))
        return Failure(
            "storage_unavailable" if temporary else "storage_invalid_response", temporary
        )
    if isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError)):
        return Failure("storage_unavailable", True)
    if isinstance(error, UnexpectedResponse):
        return Failure(
            "storage_unavailable"
            if error.status_code in {429, 500, 502, 503, 504}
            else "storage_rejected",
            error.status_code in {429, 500, 502, 503, 504},
        )
    if isinstance(error, PermissionError):
        return Failure("invalid_configuration", False)
    if isinstance(error, OSError) and not isinstance(error, FileNotFoundError):
        return Failure("storage_unavailable", True)
    return Failure("invalid_contract_or_invariant", False)
