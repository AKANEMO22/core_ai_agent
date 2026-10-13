"""Neural embeddings and an explicitly labelled offline LSA baseline."""

import hashlib
import os
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from .ontology import tokens


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def signature(self) -> str: ...

    @abstractmethod
    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray: ...


class NeuralEmbedding(EmbeddingProvider):
    def __init__(
        self, model: str = "Qwen/Qwen3-Embedding-0.6B", revision: str | None = None
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(int(os.getenv("RAG_TORCH_THREADS", "4")))
        self.name = model
        self.query_cache = OrderedDict()
        self.revision = revision
        self.model = SentenceTransformer(
            model,
            revision=revision,
            device=os.getenv("RAG_DEVICE", "cpu"),
            trust_remote_code=False,
        )
        self.model.max_seq_length = 512
        if Path(model).is_dir():
            hasher = hashlib.sha256()
            for path in sorted(Path(model).rglob("*")):
                if path.is_file() and path.suffix in {".json", ".safetensors", ".bin"}:
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            hasher.update(block)
            self.resolved_revision = None
            self.local_hash = hasher.hexdigest()[:24]
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(hf_hub_download(model, "config.json", revision=revision))
            self.resolved_revision = config_path.parent.name
            self.local_hash = None

    @property
    def signature(self):
        protocol = ":qwen-query-v1" if "qwen3-embedding" in self.name.casefold() else ""
        return f"neural:{self.name}@{self.resolved_revision or self.local_hash}:512{protocol}"

    def encode(self, texts, *, query=False):
        if not texts:
            return np.empty(
                (0, self.model.get_sentence_embedding_dimension()), dtype=np.float32
            )
        if query and len(texts) == 1:
            key = texts[0]
            if key not in self.query_cache:
                self.query_cache[key] = self._encode([key], query=True)[0]
                if len(self.query_cache) > 128:
                    self.query_cache.popitem(last=False)
            self.query_cache.move_to_end(key)
            return self.query_cache[key].copy()[None, :]
        return self._encode(texts, query=query)

    def _encode(self, texts, *, query=False):
        if "qwen3-embedding" in self.name.casefold():
            # Official Qwen protocol: instruction on queries, no prompt on documents.
            return np.asarray(
                self.model.encode(
                    texts,
                    **({"prompt_name": "query"} if query else {"prompt": ""}),
                    batch_size=4,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        prefix = "query: " if query else "passage: "
        # E5 requires asymmetric prefixes. Other models use their own model prompts.
        if "e5" in self.name.casefold():
            texts = [prefix + t for t in texts]
        method = (
            self.model.encode
            if "e5" in self.name.casefold()
            else self.model.encode_query
            if query
            else self.model.encode_document
        )
        return np.asarray(
            method(
                texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )


class LSAEmbedding(EmbeddingProvider):
    def __init__(self, state: dict | None = None):
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.components: np.ndarray | None = None
        if state:
            self.vocab = {key: int(value) for key, value in state["vocab"].items()}
            self.idf = {key: float(value) for key, value in state["idf"].items()}
            self.components = (
                np.asarray(state["components"], dtype=np.float32)
                if state["components"] is not None
                else None
            )

    @property
    def signature(self):
        return "lsa:tfidf-svd-64-v1"

    def fit(self, texts):
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds

        documents = [tokens(text) for text in texts]
        document_frequency = Counter(token for doc in documents for token in set(doc))
        vocabulary = sorted(document_frequency)
        if not vocabulary:
            raise ValueError("Cannot fit offline embeddings on an empty vocabulary")
        self.vocab = {token: index for index, token in enumerate(vocabulary)}
        count = len(documents)
        self.idf = {
            token: float(np.log((count + 1) / (frequency + 1)) + 1)
            for token, frequency in document_frequency.items()
        }

        rows, columns, values = [], [], []
        for row, doc in enumerate(documents):
            term_frequency = Counter(doc)
            for token, frequency in term_frequency.items():
                rows.append(row)
                columns.append(self.vocab[token])
                values.append(frequency * self.idf[token])
        matrix = csr_matrix(
            (np.asarray(values, dtype=np.float32), (rows, columns)),
            shape=(count, len(vocabulary)),
            dtype=np.float32,
        )

        dimensions = min(64, min(matrix.shape) - 1)
        if dimensions > 0:
            # svds requires k < min(shape); fixed random state keeps index builds reproducible.
            left, singular, right = svds(matrix, k=dimensions, random_state=42)
            order = np.argsort(singular)[::-1]
            self.components = np.asarray(right[order], dtype=np.float32)
        else:
            self.components = None
        return self._project_matrix(matrix)

    def encode(self, texts, *, query=False):
        vectors = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(tokens(text))
            for token, count in counts.items():
                if token in self.vocab:
                    vectors[row, self.vocab[token]] = count * self.idf[token]
        return self._project_matrix(vectors)

    def _project_matrix(self, matrix):
        vectors = (
            np.asarray(matrix @ self.components.T, dtype=np.float32)
            if self.components is not None
            else np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=np.float32)
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        np.divide(vectors, norms, out=vectors, where=norms > 0)
        return vectors

    def export(self):
        return {
            "vocab": self.vocab,
            "idf": self.idf,
            "components": self.components.tolist() if self.components is not None else None,
        }
