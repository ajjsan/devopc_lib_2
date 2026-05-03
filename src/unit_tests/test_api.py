import os
import sys
import unittest
from unittest.mock import patch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _root)

os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-jwt-must-be-long-enough-32"
os.environ["API_USERNAME"] = "testuser"
os.environ["API_PASSWORD"] = "testpass"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from fastapi.testclient import TestClient  # noqa: E402

import src.api as api  # noqa: E402
from src.database import init_db  # noqa: E402


class DummyModel:
    def predict(self, items):
        results = []
        for text in items:
            text_l = text.lower()
            if "happy" in text_l or "good" in text_l:
                results.append(1)
            else:
                results.append(0)
        return results


class TestAPI(unittest.TestCase):
    def setUp(self):
        init_db()
        self.client = TestClient(api.app)
        api.load_model.cache_clear()

    def tearDown(self):
        self.client.close()
        api.load_model.cache_clear()

    def _bearer_headers(self) -> dict[str, str]:
        token_resp = self.client.post(
            "/auth/token",
            data={"username": "testuser", "password": "testpass"},
        )
        self.assertEqual(token_resp.status_code, 200, token_resp.text)
        token = token_resp.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_health_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("status", body)
        self.assertIn("model_loaded", body)

    def test_token_rejects_bad_password(self):
        response = self.client.post(
            "/auth/token",
            data={"username": "testuser", "password": "wrong"},
        )
        self.assertEqual(response.status_code, 401)

    def test_predict_requires_auth(self):
        with patch("src.api.load_model", return_value=DummyModel()):
            response = self.client.post("/predict", json={"text": "hello"})
        self.assertEqual(response.status_code, 401)

    def test_predict_success(self):
        with patch("src.api.load_model", return_value=DummyModel()):
            response = self.client.post(
                "/predict",
                json={"text": "I am very happy today"},
                headers=self._bearer_headers(),
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["sentiment"], 1)
        self.assertEqual(body["label"], "positive")
        self.assertIsNotNone(body.get("prediction_id"))

    def test_predict_empty_text(self):
        with patch("src.api.load_model", return_value=DummyModel()):
            response = self.client.post(
                "/predict",
                json={"text": "   "},
                headers=self._bearer_headers(),
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("не должно быть пустым", response.json()["detail"])

    def test_predict_model_not_found(self):
        with patch("src.api.load_model", side_effect=FileNotFoundError("model not found")):
            response = self.client.post(
                "/predict",
                json={"text": "hello"},
                headers=self._bearer_headers(),
            )
        self.assertEqual(response.status_code, 500)
        self.assertIn("model not found", response.json()["detail"])

    def test_predict_batch_success(self):
        with patch("src.api.load_model", return_value=DummyModel()):
            response = self.client.post(
                "/predict-batch",
                json={"texts": ["I am happy", "I am sad"]},
                headers=self._bearer_headers(),
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["predictions"]), 2)
        self.assertEqual(body["predictions"][0]["label"], "positive")
        self.assertEqual(body["predictions"][1]["label"], "negative")
        self.assertIsNotNone(body["predictions"][0].get("prediction_id"))

    def test_predict_batch_empty_item(self):
        with patch("src.api.load_model", return_value=DummyModel()):
            response = self.client.post(
                "/predict-batch",
                json={"texts": ["ok", "   "]},
                headers=self._bearer_headers(),
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("не должно быть пустых строк", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
