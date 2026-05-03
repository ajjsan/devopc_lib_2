import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from .settings import get_settings

# tokenUrl попадает в OpenAPI → Swagger «Authorize» сам ходит на /auth/token и подставляет Bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def _constant_time_equal(actual: str, expected: str) -> bool:
    a, b = actual.encode("utf-8"), expected.encode("utf-8")
    if len(a) != len(b):
        return False
    return secrets.compare_digest(a, b)


def verify_credentials(username: str, password: str) -> bool:
    settings = get_settings()
    return _constant_time_equal(username, settings.api_username) and _constant_time_equal(
        password, settings.api_password
    )


def create_access_token(subject: str) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> str:
    settings = get_settings()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверный или просроченный токен. Нажми Authorize в Swagger и войди снова.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        username = payload.get("sub")
        if username is None or not isinstance(username, str):
            raise credentials_exception
    except JWTError as exc:
        raise credentials_exception from exc
    return username
