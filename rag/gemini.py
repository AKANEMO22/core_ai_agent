"""Gemini REST embeddings: independent inputs, bounded retries, no secret logging."""

import hashlib
import json
import os
import re
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

import httpx
import numpy as np

from .embedding import EmbeddingProvider


class GeminiEmbedding(EmbeddingProvider):
    def __init__(
        self,
        model="gemini-embedding-2",
        dimensions=768,
        *,
        api_key=None,
        cache_path=None,
        transport=None,
        sleep=time.sleep,
    ):
        if model != "gemini-embedding-2":
            raise ValueError("Gemini provider supports gemini-embedding-2 only")
        if not 128 <= dimensions <= 3072:
            raise ValueError("Gemini dimensions must be in [128,3072]")
        self.name, self.dimensions = model, dimensions
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        self.cache_path = Path(cache_path) if cache_path else None
        self.transport, self.sleep = transport, sleep
        self.query_cache = OrderedDict()
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.cache_path) as db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS embeddings(key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
                )

    @property
    def signature(self):
        # Google manages the model alias; this is not an immutable weights revision.
        return f"gemini:{self.name}:dim{self.dimensions}:search-v1:redact-v1"

    @staticmethod
    def redact(text):
        text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email]", text)
        return re.sub(r"(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)", "[number]", text)

    def _key(self, text):
        return hashlib.sha256((self.signature + "\0" + text).encode()).hexdigest()

    def encode(self, texts, *, query=False):
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        vectors = []
        with httpx.Client(
            transport=self.transport, timeout=10, follow_redirects=False
        ) as client:
            for text in texts:
                text = self.redact(text)
                formatted = (
                    f"task: search result | query: {text}"
                    if query
                    else f"title: none | text: {text}"
                )
                key, vector = self._key(formatted), None
                if query:
                    vector = self.query_cache.get(key)
                elif self.cache_path:
                    with sqlite3.connect(self.cache_path) as db:
                        row = db.execute(
                            "SELECT vector FROM embeddings WHERE key=?", (key,)
                        ).fetchone()
                        if row:
                            vector = np.asarray(json.loads(row[0]), dtype=np.float32)
                if vector is None:
                    vector = self._request(client, formatted)
                    if query:
                        self.query_cache[key] = vector
                        if len(self.query_cache) > 128:
                            self.query_cache.popitem(last=False)
                    elif self.cache_path:
                        with sqlite3.connect(self.cache_path) as db:
                            db.execute(
                                "INSERT OR REPLACE INTO embeddings VALUES(?,?)",
                                (key, json.dumps(vector.tolist())),
                            )
                if query:
                    self.query_cache.move_to_end(key)
                vectors.append(vector.copy())
        return np.asarray(vectors, dtype=np.float32)

    def _request(self, client, text):
        if not self.api_key:
            raise RuntimeError("Set GEMINI_API_KEY in the process environment")
        # One text per embedContent: Embedding 2 aggregates multiple inputs.
        body = {
            "model": "models/" + self.name,
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.dimensions,
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.name}:embedContent"
        for attempt in range(3):
            try:
                response = client.post(
                    url, headers={"x-goog-api-key": self.api_key}, json=body
                )
            except httpx.TransportError:
                if attempt == 2:
                    raise RuntimeError(
                        "Gemini embedding transport unavailable"
                    ) from None
                self.sleep(0.5 * 2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 2:
                    raise RuntimeError(
                        f"Gemini embedding unavailable (HTTP {response.status_code})"
                    )
                try:
                    delay = float(response.headers.get("Retry-After", "0"))
                except ValueError:
                    delay = 0
                self.sleep(min(5, max(0.5 * 2**attempt, delay)))
                continue
            if response.status_code != 200:
                raise RuntimeError(
                    f"Gemini embedding rejected (HTTP {response.status_code})"
                )
            try:
                vector = np.asarray(
                    response.json()["embedding"]["values"], dtype=np.float32
                )
                norm = np.linalg.norm(vector)
                if (
                    vector.shape != (self.dimensions,)
                    or not np.isfinite(vector).all()
                    or not np.isfinite(norm)
                    or norm <= 0
                ):
                    raise ValueError("Invalid embedding")
            except (ValueError, KeyError, TypeError):
                raise RuntimeError("Gemini returned an invalid embedding") from None
            return vector / norm
        raise RuntimeError("Gemini embedding unavailable")
