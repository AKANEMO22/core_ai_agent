"""JWT authentication and authorization utilities for service and candidate session tokens."""

import logging
import re
import time
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import get_settings
from core.runtime.agent import sanitize_secrets

logger = logging.getLogger(__name__)

# Security scheme for FastAPI OpenAPI docs
security_scheme = HTTPBearer(auto_error=False)


class AuthError(Exception):
    """Exception raised for authentication and token validation failures."""

    def __init__(self, message: str) -> None:
        # Sanitize message to prevent leaking secrets or raw token fragments
        clean_msg = sanitize_secrets(message)
        # Scrub potential JWT fragments (three dot-separated base64-like strings)
        clean_msg = re.sub(
            r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
            "[REDACTED_TOKEN]",
            clean_msg,
        )
        super().__init__(clean_msg)


def _scrub_token_leak(error_str: str, raw_token: Optional[str] = None, secret: Optional[str] = None) -> str:
    """Ensure error messages never leak raw token strings or secret keys."""
    clean = sanitize_secrets(error_str)
    if raw_token and raw_token in clean:
        clean = clean.replace(raw_token, "[REDACTED_TOKEN]")
    if secret and secret in clean:
        clean = clean.replace(secret, "[REDACTED_SECRET]")
    clean = re.sub(
        r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
        "[REDACTED_TOKEN]",
        clean,
    )
    return clean


def create_service_token(
    issuer: str = "backend",
    audience: str = "ai-runtime",
    secret: Optional[str] = None,
    expires_minutes: Optional[int] = None,
) -> str:
    """Generate a signed HMAC SHA-256 JWT service token for backend service communication.

    Args:
        issuer: Token issuer claim ('iss').
        audience: Target service audience claim ('aud').
        secret: Signing key secret (defaults to Settings.JWT_SECRET).
        expires_minutes: Token validity window in minutes (defaults to Settings.SERVICE_TOKEN_EXPIRE_MINUTES).

    Returns:
        str: Encoded JWT token string.
    """
    settings = get_settings()
    signing_secret = secret if secret is not None else settings.JWT_SECRET
    exp_minutes = (
        expires_minutes
        if expires_minutes is not None
        else settings.SERVICE_TOKEN_EXPIRE_MINUTES
    )

    now = int(time.time())
    payload: Dict[str, Any] = {
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "exp": now + int(exp_minutes * 60),
    }

    return jwt.encode(payload, signing_secret, algorithm=settings.JWT_ALGORITHM)


def verify_service_token(
    token: str,
    secret: Optional[str] = None,
    expected_audience: str = "ai-runtime",
    expected_issuer: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a backend service token verifying signature, audience, expiration, and issuer.

    Args:
        token: Raw JWT token string.
        secret: Verification key secret (defaults to Settings.JWT_SECRET).
        expected_audience: Expected audience claim ('aud').
        expected_issuer: Optional expected issuer claim ('iss').

    Returns:
        Dict[str, Any]: Validated token claims dictionary.

    Raises:
        AuthError: If signature, audience, expiry, or issuer validation fails.
    """
    settings = get_settings()
    signing_secret = secret if secret is not None else settings.JWT_SECRET

    try:
        claims = jwt.decode(
            token,
            signing_secret,
            algorithms=[settings.JWT_ALGORITHM],
            audience=expected_audience,
            options={
                "verify_signature": True,
                "verify_aud": True,
                "verify_exp": True,
                "require": ["exp", "iss", "aud"],
            },
        )
    except jwt.ExpiredSignatureError as e:
        raise AuthError(f"Service token has expired: {_scrub_token_leak(str(e), token, signing_secret)}") from e
    except jwt.InvalidAudienceError as e:
        raise AuthError(f"Invalid service token audience: {_scrub_token_leak(str(e), token, signing_secret)}") from e
    except jwt.InvalidIssuerError as e:
        raise AuthError(f"Invalid service token issuer: {_scrub_token_leak(str(e), token, signing_secret)}") from e
    except jwt.PyJWTError as e:
        raise AuthError(f"Invalid service token signature or format: {_scrub_token_leak(str(e), token, signing_secret)}") from e
    except Exception as e:
        raise AuthError(f"Service token verification failed: {_scrub_token_leak(str(e), token, signing_secret)}") from e

    if expected_issuer is not None and claims.get("iss") != expected_issuer:
        raise AuthError(
            f"Token issuer mismatch: expected '{expected_issuer}', got '{claims.get('iss')}'"
        )

    return claims


def create_session_token(
    session_id: str,
    audience: str = "ai-interview-session",
    secret: Optional[str] = None,
    expires_minutes: Optional[int] = None,
    **extra_claims: Any,
) -> str:
    """Generate a scoped JWT token for a candidate WebSocket or REST session.

    Args:
        session_id: Unique session identifier scoped to this token.
        audience: Target audience claim (default 'ai-interview-session').
        secret: Signing key secret (defaults to Settings.JWT_SECRET).
        expires_minutes: Validity window in minutes (defaults to Settings.SESSION_TOKEN_EXPIRE_MINUTES).
        **extra_claims: Additional custom claims to embed in payload.

    Returns:
        str: Encoded JWT token string.
    """
    settings = get_settings()
    signing_secret = secret if secret is not None else settings.JWT_SECRET
    exp_minutes = (
        expires_minutes
        if expires_minutes is not None
        else settings.SESSION_TOKEN_EXPIRE_MINUTES
    )

    now = int(time.time())
    payload: Dict[str, Any] = {
        "sub": session_id,
        "session_id": session_id,
        "aud": audience,
        "iat": now,
        "exp": now + int(exp_minutes * 60),
    }
    payload.update(extra_claims)

    return jwt.encode(payload, signing_secret, algorithm=settings.JWT_ALGORITHM)


def verify_session_token(
    token: str,
    secret: Optional[str] = None,
    expected_session_id: Optional[str] = None,
    expected_audience: str = "ai-interview-session",
) -> Dict[str, Any]:
    """Validate a candidate session token verifying signature, audience, expiry, and session scope.

    Args:
        token: Raw JWT token string.
        secret: Verification key secret (defaults to Settings.JWT_SECRET).
        expected_session_id: Optional session_id that the token must be scoped to.
        expected_audience: Expected audience claim (default 'ai-interview-session').

    Returns:
        Dict[str, Any]: Validated token claims dictionary.

    Raises:
        AuthError: If validation fails or token is not scoped to expected_session_id.
    """
    settings = get_settings()
    signing_secret = secret if secret is not None else settings.JWT_SECRET

    try:
        claims = jwt.decode(
            token,
            signing_secret,
            algorithms=[settings.JWT_ALGORITHM],
            audience=expected_audience,
            options={
                "verify_signature": True,
                "verify_aud": True,
                "verify_exp": True,
                "require": ["exp", "aud"],
            },
        )
    except jwt.ExpiredSignatureError as e:
        raise AuthError(f"Session token has expired: {_scrub_token_leak(str(e), token, signing_secret)}") from e
    except jwt.InvalidAudienceError as e:
        raise AuthError(f"Invalid session token audience: {_scrub_token_leak(str(e), token, signing_secret)}") from e
    except jwt.PyJWTError as e:
        raise AuthError(f"Invalid session token signature or format: {_scrub_token_leak(str(e), token, signing_secret)}") from e
    except Exception as e:
        raise AuthError(f"Session token verification failed: {_scrub_token_leak(str(e), token, signing_secret)}") from e

    scoped_session = claims.get("session_id") or claims.get("sub")
    if expected_session_id is not None and scoped_session != expected_session_id:
        raise AuthError(
            f"Token is not authorized for session '{expected_session_id}' (token scope: '{scoped_session}')"
        )

    return claims


async def require_service_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
) -> Dict[str, Any]:
    """FastAPI dependency requiring a valid Bearer service token in Authorization header."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return verify_service_token(credentials.credentials)
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def require_session_or_service_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
) -> Dict[str, Any]:
    """FastAPI dependency accepting either a valid service token or a session token."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    # Try service token first
    try:
        return verify_service_token(token)
    except AuthError:
        pass

    # Fallback to session token
    try:
        return verify_session_token(token)
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service or session token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
