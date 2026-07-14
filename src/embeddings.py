"""Embedding extraction via OpenRouter API with parallel support."""

import os
import json
import time
import threading
import numpy as np
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import httpx


# OpenRouter models configuration
MODELS = {
    "openai-small": {
        "provider": "openrouter",
        "model_id": "openai/text-embedding-3-small",
        "dimensions": 1536,
        "parameters": None,
        "type": "commercial",
    },
    "openai-large": {
        "provider": "openrouter",
        "model_id": "openai/text-embedding-3-large",
        "dimensions": 3072,
        "parameters": None,
        "type": "commercial",
    },
    "gemini-001": {
        "provider": "openrouter",
        "model_id": "google/gemini-embedding-001",
        "dimensions": 768,
        "parameters": None,
        "type": "commercial",
    },
    "mistral-embed": {
        "provider": "openrouter",
        "model_id": "mistralai/mistral-embed-2312",
        "dimensions": 1024,
        "parameters": None,
        "type": "commercial",
    },
    "qwen3-8b": {
        "provider": "openrouter",
        "model_id": "qwen/qwen3-embedding-8b",
        "dimensions": 4096,
        "parameters": 8_000_000_000,
        "type": "open-source",
    },
    "qwen3-4b": {
        "provider": "openrouter",
        "model_id": "qwen/qwen3-embedding-4b",
        "dimensions": 2560,
        "parameters": 4_000_000_000,
        "type": "open-source",
    },
    "bge-m3": {
        "provider": "openrouter",
        "model_id": "baai/bge-m3",
        "dimensions": 1024,
        "parameters": 568_000_000,
        "type": "open-source",
    },
    "multilingual-e5-large": {
        "provider": "openrouter",
        "model_id": "intfloat/multilingual-e5-large",
        "dimensions": 1024,
        "parameters": 560_000_000,
        "type": "open-source",
    },
    "gemini-2": {
        "provider": "openrouter",
        "model_id": "google/gemini-embedding-2",
        "dimensions": 3072,
        "parameters": None,
        "type": "commercial",
    },
}


class EmbeddingExtractor:
    """Extract embeddings from multiple providers via OpenRouter."""

    def __init__(self, api_key: Optional[str] = None, max_workers: int = 5):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.max_workers = max_workers
        self._locks = {model: threading.Lock() for model in MODELS}

    def _get_embeddings_httpx(
        self,
        texts: list[str],
        model_id: str,
    ) -> list[list[float]]:
        """Fallback: get embeddings via httpx when OpenAI SDK fails."""
        resp = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model_id, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        return [item["embedding"] for item in body["data"]]

    def get_embeddings(
        self,
        texts: list[str],
        model_key: str,
        batch_size: int = 100,
    ) -> np.ndarray:
        """Get embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            model_key: Key from MODELS dict.
            batch_size: Number of texts per API call.

        Returns:
            numpy array of shape (len(texts), dimensions)
        """
        model_config = MODELS[model_key]
        model_id = model_config["model_id"]
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=model_id,
                        input=batch,
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break
                except ValueError:
                    # OpenAI SDK parse failure (e.g. gemini-embedding-2)
                    try:
                        batch_embeddings = self._get_embeddings_httpx(batch, model_id)
                        all_embeddings.extend(batch_embeddings)
                        break
                    except Exception as e2:
                        if attempt < max_retries - 1:
                            wait = 2 ** (attempt + 1)
                            print(f"  Retry {attempt+1}/{max_retries} for {model_key}: {e2}")
                            time.sleep(wait)
                        else:
                            raise RuntimeError(
                                f"Failed to get embeddings for {model_key} after {max_retries} attempts: {e2}"
                            )
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        print(f"  Retry {attempt+1}/{max_retries} for {model_key}: {e}")
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"Failed to get embeddings for {model_key} after {max_retries} attempts: {e}"
                        )

            if i + batch_size < len(texts):
                time.sleep(0.2)

        return np.array(all_embeddings, dtype=np.float32)

    def get_embeddings_parallel(
        self,
        tasks: dict[str, list[str]],
        batch_size: int = 100,
    ) -> dict[str, np.ndarray]:
        """Extract embeddings for multiple models in parallel.

        Args:
            tasks: {model_key: [texts]} for each model.
            batch_size: Number of texts per API call.

        Returns:
            {model_key: np.ndarray} with embeddings for each model.
        """
        results = {}

        def _extract(composite_key: str, texts: list[str], delay: float = 0) -> tuple[str, np.ndarray]:
            base_model = composite_key.rsplit("_", 1)[0]
            if base_model not in MODELS:
                base_model = composite_key.rsplit("_", 2)[0]
            time.sleep(delay)
            with self._locks[base_model]:
                return composite_key, self.get_embeddings(texts, base_model, batch_size)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_extract, key, texts, i * 0.5): key
                for i, (key, texts) in enumerate(tasks.items())
            }
            for future in as_completed(futures):
                key, embeddings = future.result()
                results[key] = embeddings

        return results

    def get_embedding(self, text: str, model_key: str) -> np.ndarray:
        """Get embedding for a single text."""
        result = self.get_embeddings([text], model_key)
        return result[0]

    def get_model_info(self, model_key: str) -> dict:
        """Get configuration info for a model."""
        return MODELS[model_key].copy()

    def list_models(self) -> list[str]:
        """List all available model keys."""
        return list(MODELS.keys())

    def get_sentence_embeddings(
        self,
        template: str,
        words: list[str],
        model_key: str,
    ) -> np.ndarray:
        """Get embeddings for sentences formed by filling template with words.

        Args:
            template: Sentence template with {word} placeholder.
            words: List of words to fill into the template.
            model_key: Key from MODELS dict.

        Returns:
            numpy array of shape (len(words), dimensions)
        """
        sentences = [template.format(word=w) for w in words]
        return self.get_embeddings(sentences, model_key)

    def get_seat_embeddings(
        self,
        seat_config: dict,
        model_key: str,
    ) -> dict[str, np.ndarray]:
        """Get all embeddings needed for a SEAT test.

        Args:
            seat_config: SEAT test configuration from weat_ptbr.json.
            model_key: Key from MODELS dict.

        Returns:
            dict with 'target1', 'target2', 'attribute1', 'attribute2' embeddings.
        """
        template1 = seat_config["target1_template"]
        template2 = seat_config["target2_template"]

        target1_words = seat_config["targets"][seat_config.get("target1_label", list(seat_config["targets"].keys())[0])]
        target2_words = seat_config["targets"][seat_config.get("target2_label", list(seat_config["targets"].keys())[1])]

        target1_embs = self.get_sentence_embeddings(template1, target1_words, model_key)
        target2_embs = self.get_sentence_embeddings(template2, target2_words, model_key)
        attr1_embs = self.get_embeddings(seat_config["attribute1"]["words"], model_key)
        attr2_embs = self.get_embeddings(seat_config["attribute2"]["words"], model_key)

        return {
            "target1": target1_embs,
            "target2": target2_embs,
            "attribute1": attr1_embs,
            "attribute2": attr2_embs,
        }


def load_weat_lists(filepath: str) -> dict:
    """Load WEAT word lists from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
