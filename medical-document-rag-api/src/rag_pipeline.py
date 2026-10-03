"""Base RAG pipeline for medical-document question answering.

This file is intentionally simple and educational. It shows the main project flow:
text extraction result -> chunks -> embeddings -> retrieval -> grounded answer.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from .config import EMBEDDING_MODEL, CHAT_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class Chunk:
    text: str
    page: int | None = None
    title: str = "Uploaded medical document"
    source: str = "local upload"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """Split document text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(text=chunk))
        start += chunk_size - overlap
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    """Create embeddings for a list of texts."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Normalise vectors for cosine-similarity search."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class MedicalRAG:
    """Small in-memory RAG pipeline for learning and prototyping."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.chunk_vectors: np.ndarray | None = None

    def index_text(self, text: str) -> None:
        """Chunk and embed extracted document text."""
        self.chunks = chunk_text(text)
        self.chunk_vectors = normalise(embed_texts([chunk.text for chunk in self.chunks]))

    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks for a question."""
        if self.chunk_vectors is None or not self.chunks:
            raise ValueError("No document has been indexed yet.")

        question_vector = normalise(embed_texts([question]))[0]
        scores = self.chunk_vectors @ question_vector
        best_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for index in best_indices:
            chunk = self.chunks[index]
            results.append({
                "text": chunk.text,
                "page": chunk.page,
                "title": chunk.title,
                "source": chunk.source,
                "score": float(scores[index]),
            })
        return results

    def answer_question(self, question: str, top_k: int = TOP_K) -> str:
        """Generate a grounded answer using retrieved chunks."""
        retrieved = self.retrieve(question, top_k=top_k)
        context = "\n\n".join(
            f"[Source {i + 1}] {item['title']} | page={item['page']} | {item['source']}\n{item['text']}"
            for i, item in enumerate(retrieved)
        )

        system_prompt = (
            "You answer using only the supplied medical document sources. "
            "If evidence is missing, say so. Do not diagnose, prescribe or make a personalised treatment decision. "
            "Use cautious language, cite source numbers and recommend consulting a qualified healthcare professional for personal medical concerns."
        )
        user_prompt = f"Question: {question}\n\nSources:\n{context}"

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
