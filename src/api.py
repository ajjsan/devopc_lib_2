import configparser
import os
import pickle
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import auth
from .database import get_db, init_db
from .db_models import PredictionRecord

CONFIG_PATH = os.path.join(os.getcwd(), "config.ini")


def get_model_path() -> str:
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH, encoding="utf-8")
    return os.path.normpath(
        config.get("LOG_REG", "model_path", fallback=os.path.join("experiments", "tfidf_log_reg.pkl"))
    )


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст твита")


class PredictResponse(BaseModel):
    sentiment: int
    label: str
    prediction_id: int | None = Field(default=None, description="ID записи в БД")


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Список текстов для батч-предсказания")


class PredictBatchResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_path: str
    model_loaded: bool


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Twitter Sentiment API",
    description=(
        "Классификация тональности (TF-IDF + LogisticRegression). "
        "**Swagger:** нажми **Authorize**, выбери OAuth2 Password, укажи `username` и `password` "
        "как в `.env` (`API_USERNAME`, `API_PASSWORD`). После входа заголовок `Authorization` "
        "подставится сам для `/predict` и `/predict-batch`. "
        "Поле **client_id** можно оставить пустым."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@lru_cache(maxsize=1)
def load_model():
    model_path = get_model_path()
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Модель не найдена по пути '{model_path}'. Сначала запусти обучение: python .\\src\\train.py"
        )

    with open(model_path, "rb") as f:
        return pickle.load(f)


def sentiment_to_label(sentiment: int) -> str:
    return "positive" if sentiment == 1 else "negative"


@app.get("/", tags=["service"])
def root():
    return {
        "message": "Twitter Sentiment API is running",
        "auth": {
            "swagger": "GET /docs → Authorize → OAuth2 Password: username/password из .env (API_USERNAME, API_PASSWORD)",
            "curl": "POST /auth/token + заголовок Authorization: Bearer <access_token> для /predict",
        },
    }


@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if not auth.verify_credentials(form_data.username, form_data.password):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")
    token = auth.create_access_token(form_data.username)
    return TokenResponse(access_token=token)


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health_check():
    try:
        load_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False

    return HealthResponse(
        status="ok",
        model_path=get_model_path(),
        model_loaded=model_loaded,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict_sentiment(
    payload: PredictRequest,
    db: Session = Depends(get_db),
    _user: str = Depends(auth.get_current_user),
):
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Поле text не должно быть пустым")

    sentiment = int(model.predict([text])[0])
    label = sentiment_to_label(sentiment)
    row = PredictionRecord(
        input_text=text,
        sentiment=sentiment,
        label=label,
        batch_index=None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return PredictResponse(sentiment=sentiment, label=label, prediction_id=row.id)


@app.post("/predict-batch", response_model=PredictBatchResponse, tags=["inference"])
def predict_batch(
    payload: PredictBatchRequest,
    db: Session = Depends(get_db),
    _user: str = Depends(auth.get_current_user),
):
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    cleaned_texts = [text.strip() for text in payload.texts]
    if any(not text for text in cleaned_texts):
        raise HTTPException(status_code=400, detail="В поле texts не должно быть пустых строк")

    sentiments = model.predict(cleaned_texts)
    predictions: list[PredictResponse] = []
    for idx, (text, sentiment_raw) in enumerate(zip(cleaned_texts, sentiments, strict=True)):
        sentiment = int(sentiment_raw)
        label = sentiment_to_label(sentiment)
        row = PredictionRecord(
            input_text=text,
            sentiment=sentiment,
            label=label,
            batch_index=idx,
        )
        db.add(row)
        db.flush()
        predictions.append(PredictResponse(sentiment=sentiment, label=label, prediction_id=row.id))
    db.commit()
    return PredictBatchResponse(predictions=predictions)
