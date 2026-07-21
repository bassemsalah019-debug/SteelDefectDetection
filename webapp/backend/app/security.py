"""Password hashing (argon2) and JWT access/refresh tokens."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import get_settings

settings = get_settings()
_pwd = CryptContext(schemes=["argon2"], deprecated="auto")
_ALGO = "HS256"


def hash_password(password: str) -> str:
    return _pwd.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return _pwd.verify(password, hashed)


def _make_token(sub: str, ttl: timedelta, token_type: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": sub, "type": token_type, "iat": now, "exp": now + ttl}
    return jwt.encode(payload, settings.secret_key, algorithm=_ALGO)


def create_access_token(sub: str) -> str:
    return _make_token(sub, timedelta(minutes=settings.access_token_expire_minutes), "access")


def create_refresh_token(sub: str) -> str:
    return _make_token(sub, timedelta(days=settings.refresh_token_expire_days), "refresh")


def decode_token(token: str, expected_type: str) -> str | None:
    """Return the subject (user id) if the token is valid and of the expected type."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[_ALGO])
    except JWTError:
        return None
    if payload.get("type") != expected_type:
        return None
    return payload.get("sub")
