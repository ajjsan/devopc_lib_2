#!/usr/bin/env python3
"""
Функциональные проверки HTTP API без внешних зависимостей (stdlib).
Использование:
  python scripts/functional_test_api.py --base-url http://127.0.0.1:8000 --timeout 120

Учётные данные для /auth/token: переменные окружения API_USERNAME и API_PASSWORD
(как в .env), либо аргументы --api-user / --api-password.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Optional, Tuple


def http_json(
    method: str,
    url: str,
    body: Optional[Dict] = None,
    timeout: int = 30,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict]:
    data = None
    hdrs = {"Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        data = payload
        hdrs["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.getcode()
            return status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"detail": raw}
        return exc.code, payload


def _is_transient_http_error(exc: BaseException) -> bool:
    """Сразу после docker compose порт может быть открыт, а uvicorn ещё нет — обрыв без ответа."""
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        return any(
            s in msg
            for s in (
                "remote end closed",
                "connection refused",
                "connection reset",
                "broken pipe",
                "timed out",
            )
        )
    return False


def http_form(
    url: str,
    form: Dict[str, str],
    timeout: int = 30,
) -> Tuple[int, Dict]:
    data = urllib.parse.urlencode(form).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.getcode(), json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"detail": raw}
        return exc.code, payload


def wait_for_model(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_detail = None
    health_url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            status, body = http_json("GET", health_url, timeout=15)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            if not _is_transient_http_error(exc):
                raise
            last_detail = {"transient_error": str(exc)}
            time.sleep(2)
            continue
        if status == 200 and body.get("model_loaded") is True:
            return
        last_detail = body
        time.sleep(2)
    raise RuntimeError(f"/health: model not ready after {timeout_s}s. Last response: {last_detail}")


def obtain_token(base_url: str, username: str, password: str, timeout: int) -> str:
    status, body = http_form(
        f"{base_url.rstrip('/')}/auth/token",
        {"username": username, "password": password},
        timeout=timeout,
    )
    if status != 200:
        raise RuntimeError(f"/auth/token failed: {status} {body}")
    token = body.get("access_token")
    if not token:
        raise RuntimeError(f"/auth/token: no access_token in {body}")
    return str(token)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=120, help="Ожидание готовности модели, секунды")
    parser.add_argument(
        "--api-user",
        default=os.environ.get("API_USERNAME", ""),
        help="Логин для OAuth2 (по умолчанию env API_USERNAME)",
    )
    parser.add_argument(
        "--api-password",
        default=os.environ.get("API_PASSWORD", ""),
        help="Пароль для OAuth2 (по умолчанию env API_PASSWORD)",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    if not args.api_user or not args.api_password:
        print("FAIL: задай API_USERNAME и API_PASSWORD в окружении или --api-user / --api-password", file=sys.stderr)
        return 1

    wait_for_model(base, args.timeout)

    # Ещё одна попытка после ожидания — иногда первый «успешный» /health сразу даёт сбой на следующем запросе
    for attempt in range(3):
        try:
            status, health = http_json("GET", f"{base}/health", timeout=30)
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            if attempt == 2 or not _is_transient_http_error(exc):
                raise
            time.sleep(2)
    assert status == 200, health
    assert health.get("model_loaded") is True, health

    token = obtain_token(base, args.api_user, args.api_password, timeout=30)
    auth_headers = {"Authorization": f"Bearer {token}"}

    status, pred = http_json(
        "POST",
        f"{base}/predict",
        {"text": "I am very happy today"},
        headers=auth_headers,
    )
    assert status == 200, pred
    assert pred.get("sentiment") in (0, 1), pred

    status, batch = http_json(
        "POST",
        f"{base}/predict-batch",
        {"texts": ["I love this", "This is terrible"]},
        headers=auth_headers,
    )
    assert status == 200, batch
    preds = batch.get("predictions", [])
    assert len(preds) == 2, batch

    print("OK: functional API checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
