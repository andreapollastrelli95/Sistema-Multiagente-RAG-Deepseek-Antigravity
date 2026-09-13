"""
rag_demo.py - Sistema RAG Avanzato ad Alta Precisione & Multi-Agente
Modello LLM: deepseek-r1:8b (Ollama)
Modello Embeddings: nomic-embed-text (Ollama) + Motore BM25 (Hybrid Search)
Supporto: Documenti .txt, .pdf, .md

Caratteristiche Architetturali Avanzate:
1. Hybrid Search & RRF Reranking (Ricerca Semantica Vettoriale + Ricerca Lessicale BM25)
2. Chunking Semantico (Suddivisione basata su paragrafi, sezioni logiche e confini di frase)
3. Architettura Multi-Agente:
   - Agente 1 (Ricercatore): Retrieval ibrido, contestualizzazione ed estrazione passaggi
   - Agente 2 (Validatore/Revisore): Verifica anti-allucinazione e conformità citazioni fonti
4. Mantenimento della Memoria (Chat Memory) per conversazioni multi-turno
"""

import os
import sys
import re
import math
import json
import time
import hashlib
import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Set

# Assicura il corretto encoding UTF-8 sulla console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Sopprimi log e warning interni di pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)

import requests
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN_SPLITTER = True
except ImportError:
    HAS_LANGCHAIN_SPLITTER = False

console = Console(highlight=False)

# Configurazioni di default
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "deepseek-r1:8b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
CACHE_FILE = ".rag_cache.json"
IGNORED_DIRS = {".git", ".gemini", "venv", ".venv", "__pycache__", "site-packages", ".agents"}


# =====================================================================
# 1. MODELLO DATI (DOCUMENT CHUNK & RISULTATI)
# =====================================================================

class DocumentChunk:
    """Rappresenta un singolo frammento di testo con i suoi metadati strutturati."""
    def __init__(
        self,
        text: str,
        source: str,
        page: Optional[int] = None,
        chunk_id: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text.strip()
        self.source = source
        self.page = page
        self.chunk_id = chunk_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(
            text=data["text"],
            source=data["source"],
            page=data.get("page"),
            chunk_id=data.get("chunk_id", 0),
            metadata=data.get("metadata", {}),
        )


class SearchResult:
    """Rappresenta il risultato di una ricerca ibrida con punteggi dettagliati."""
    def __init__(
        self,
        chunk: DocumentChunk,
        hybrid_score: float,
        dense_score: float = 0.0,
        bm25_score: float = 0.0,
        rrf_score: float = 0.0
    ):
        self.chunk = chunk
        self.hybrid_score = hybrid_score
        self.dense_score = dense_score
        self.bm25_score = bm25_score
        self.rrf_score = rrf_score


# =====================================================================
# 2. CHUNKING SEMANTICO (SEMANTIC TEXT SPLITTER)
# =====================================================================

class SemanticTextSplitter:
    """
    Suddivide il testo in chunk logici rispettando i confini di:
    - Sezioni e titoli Markdown / numerati (#, ##, 1., 2.)
    - Paragrafi (doppio newline)
    - Punti elenco e frasi complete
    Evita rigorosamente tagli a metà parola o a metà frase.
    """

    def __init__(
        self,
        target_chunk_size: int = 700,
        min_chunk_size: int = 120,
        overlap_sentences: int = 1
    ):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min(min_chunk_size, max(40, target_chunk_size // 4))
        self.overlap_sentences = overlap_sentences

    @staticmethod
    def _split_into_sections_and_paragraphs(text: str) -> List[str]:
        """Divide il testo in blocchi coerenti di paragrafi e titoli."""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        # Riconosce intestazioni Markdown o paragrafi separati da righe vuote
        pattern = r"\n{2,}|\n(?=[#]{1,6}\s|[0-9]+\.\s+[A-Z])"
        raw_blocks = re.split(pattern, normalized)
        blocks = []
        for b in raw_blocks:
            cleaned = b.strip()
            if cleaned:
                blocks.append(cleaned)
        return blocks

    @staticmethod
    def _is_header(block: str) -> bool:
        """Verifica se il blocco rappresenta un titolo di sezione o capitolo."""
        return block.startswith("#") or bool(re.match(r"^[0-9]+\.\s+[A-Z]", block))

    @staticmethod
    def _split_into_sentences(paragraph: str) -> List[str]:
        """Divide un paragrafo in frasi preservando punteggiatura e codici/abbreviazioni."""
        # Regex avanzata che non spezza su 'es.', 'v1.2', 'SN-900' o numeri decimali
        sentence_end = re.compile(
            r"(?<!\be\.g)(?<!\bes)(?<!\bvs)(?<!\bv\d)(?<!\bpag)(?<=[.?!])\s+(?=[A-Z0-9\"\'])"
        )
        sentences = sentence_end.split(paragraph)
        return [s.strip() for s in sentences if s.strip()]

    def split_text(self, text: str) -> List[str]:
        """Esegue il chunking semantico strutturato."""
        if not text or not text.strip():
            return []

        blocks = self._split_into_sections_and_paragraphs(text)
        chunks: List[str] = []
        current_chunk_parts: List[str] = []
        current_len = 0

        for block in blocks:
            block_len = len(block)
            is_hdr = self._is_header(block)

            # Se si incontra una nuova sezione/titolo e abbiamo già accumulato testo valido, chiudi il blocco precedente
            if is_hdr and current_chunk_parts and current_len >= self.min_chunk_size:
                chunks.append("\n\n".join(current_chunk_parts).strip())
                current_chunk_parts = []
                current_len = 0

            # Se il singolo blocco è già superiore alla dimensione target, lo dividiamo per frasi
            if block_len > self.target_chunk_size:
                sentences = self._split_into_sentences(block)
                for sentence in sentences:
                    s_len = len(sentence)
                    if current_len + s_len + 1 > self.target_chunk_size and current_chunk_parts:
                        chunks.append(" ".join(current_chunk_parts).strip())
                        # Mantieni le ultime frasi come sovrapposizione semantica
                        current_chunk_parts = (
                            current_chunk_parts[-self.overlap_sentences:]
                            if self.overlap_sentences > 0 else []
                        )
                        current_len = sum(len(p) + 1 for p in current_chunk_parts)

                    current_chunk_parts.append(sentence)
                    current_len += s_len + 1
            else:
                # Accumula blocchi di paragrafi mantenendoli integri
                if current_len + block_len + 2 > self.target_chunk_size and current_chunk_parts:
                    chunks.append("\n\n".join(current_chunk_parts).strip())
                    if self.overlap_sentences > 0 and len(current_chunk_parts[-1]) < (self.target_chunk_size // 2) and not self._is_header(current_chunk_parts[-1]):
                        current_chunk_parts = [current_chunk_parts[-1]]
                        current_len = len(current_chunk_parts[0]) + 2
                    else:
                        current_chunk_parts = []
                        current_len = 0

                current_chunk_parts.append(block)
                current_len += block_len + 2

        if current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts).strip())

        # Filtra eventuali chunk residui microscopici
        valid_chunks = [c for c in chunks if len(c.strip()) >= self.min_chunk_size or len(chunks) == 1]
        return valid_chunks if valid_chunks else [text.strip()]


# Manteniamo la classe TextSplitter compatibile come alias avanzato
class TextSplitter(SemanticTextSplitter):
    """Wrapper di retrocompatibilità per SemanticTextSplitter."""
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        super().__init__(target_chunk_size=chunk_size, min_chunk_size=120, overlap_sentences=1)


# =====================================================================
# 3. DOCUMENT LOADER
# =====================================================================

class DocumentLoader:
    """Gestisce la scansione e l'estrazione del testo da .txt, .pdf e .md."""

    @staticmethod
    def is_ignored(path: Path) -> bool:
        for part in path.parts:
            if part in IGNORED_DIRS or part.startswith("."):
                return True
        return False

    @classmethod
    def load_file(cls, file_path: Path) -> List[Tuple[str, Optional[int]]]:
        """Estrae il testo da un singolo file. Ritorna lista di (testo, numero_pagina)."""
        suffix = file_path.suffix.lower()
        results = []

        if suffix in [".txt", ".md"]:
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    if content.strip():
                        results.append((content, None))
                    break
                except UnicodeDecodeError:
                    continue

        elif suffix == ".pdf":
            if not HAS_PYPDF:
                console.print(f"[yellow]Attenzione: pypdf non è installato. Salto {file_path.name}[/yellow]")
                return []
            try:
                reader = PdfReader(str(file_path))
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        results.append((page_text, i + 1))
            except Exception as e:
                console.print(f"[yellow]Errore nella lettura del PDF {file_path.name}: {e}[/yellow]")

        return results

    @classmethod
    def scan_directory(cls, directories: List[str]) -> Dict[str, List[Tuple[str, Optional[int]]]]:
        """Scansiona una o più cartelle per file supportati."""
        docs = {}
        for d in directories:
            p = Path(d)
            if not p.exists():
                continue
            if p.is_file():
                files = [p]
            else:
                files = list(p.rglob("*"))

            for file_path in files:
                if file_path.is_file() and not cls.is_ignored(file_path):
                    if file_path.suffix.lower() in [".txt", ".pdf", ".md"]:
                        content_list = cls.load_file(file_path)
                        if content_list:
                            docs[str(file_path)] = content_list
        return docs


# =====================================================================
# 4. MOTORE BM25 (RICERCA LESSICALE PER TERMINI ESATTI E CODICI)
# =====================================================================

class BM25Index:
    """
    Implementazione pura, veloce e robusta dell'algoritmo BM25Okapi.
    Specializzata per preservare codici tecnici (es. SN-900, v1.2.0),
    date (2026-09-13), acronimi (IoT, RAG) e numeri.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self.corpus_size: int = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizzazione avanzata che preserva pattern tecnici:
        - Codici alfanumerici con trattini/punti (es. SN-900, v1.2.0, ISO-9001)
        - Date e orari (es. 2026-09-13, 11434)
        - Parole standard normalizzate in lowercase
        """
        pattern = r"[a-zA-Z0-9]+(?:[-._/][a-zA-Z0-9]+)*"
        tokens = re.findall(pattern, text.lower())
        return tokens

    def fit(self, corpus: List[str]):
        """Costruisce l'indice BM25 a partire dalla lista di chunk testuali."""
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            return

        self.doc_len = []
        self.doc_freqs = []
        df: Counter = Counter()

        for doc in corpus:
            tokens = self.tokenize(doc)
            self.doc_len.append(len(tokens))
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for term in freqs:
                df[term] += 1

        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 1.0
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query: str) -> np.ndarray:
        """Calcola i punteggi BM25 di ciascun documento rispetto alla query."""
        if self.corpus_size == 0:
            return np.zeros(0, dtype=np.float32)

        query_tokens = self.tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            for idx in range(self.corpus_size):
                tf = self.doc_freqs[idx].get(token, 0)
                if tf > 0:
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[idx] / self.avgdl))
                    scores[idx] += idf * (numerator / denominator)

        return scores


# =====================================================================
# 5. VECTOR STORE CON HYBRID SEARCH E RRF RERANKING
# =====================================================================

class VectorStore:
    """
    Gestisce l'indicizzazione multimodale con:
    - Embedding vettoriali densi (Ollama / nomic-embed-text o TF-IDF)
    - Indice BM25 lessicale per codici e date
    - Reciprocal Rank Fusion (RRF) & Reranking ponderato per un'accuratezza senza perdite
    """

    def __init__(self, ollama_url: str = DEFAULT_OLLAMA_URL, embed_model: str = DEFAULT_EMBED_MODEL):
        self.ollama_url = ollama_url.rstrip("/")
        self.embed_model = embed_model
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: BM25Index = BM25Index()
        self.use_tfidf_fallback = False
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Chiama l'endpoint embeddings di Ollama."""
        try:
            res = requests.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.embed_model, "input": text},
                timeout=20
            )
            if res.status_code == 200:
                data = res.json()
                if "embeddings" in data and len(data["embeddings"]) > 0:
                    return data["embeddings"][0]

            res = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=20
            )
            if res.status_code == 200:
                data = res.json()
                if "embedding" in data:
                    return data["embedding"]
        except Exception:
            pass
        return None

    def _calculate_docs_hash(self, docs: Dict[str, List[Tuple[str, Optional[int]]]]) -> str:
        hasher = hashlib.md5()
        for fpath in sorted(docs.keys()):
            hasher.update(fpath.encode())
            try:
                mtime = os.path.getmtime(fpath)
                hasher.update(str(mtime).encode())
            except OSError:
                pass
        return hasher.hexdigest()

    def build(self, docs: Dict[str, List[Tuple[str, Optional[int]]]], force_reload: bool = False):
        """Costruisce sia l'indice semantico vettoriale sia l'indice lessicale BM25."""
        splitter = SemanticTextSplitter()
        all_chunks: List[DocumentChunk] = []
        chunk_id = 0

        for fpath, contents in docs.items():
            for text, page in contents:
                raw_chunks = splitter.split_text(text)
                for c in raw_chunks:
                    all_chunks.append(
                        DocumentChunk(
                            text=c,
                            source=fpath,
                            page=page,
                            chunk_id=chunk_id,
                            metadata={"char_len": len(c)}
                        )
                    )
                    chunk_id += 1

        self.chunks = all_chunks
        if not self.chunks:
            self.embeddings = None
            return

        # Costruisci SEMPRE l'indice BM25 (velocissimo, richiede < 10ms)
        corpus_texts = [c.text for c in self.chunks]
        self.bm25.fit(corpus_texts)

        docs_hash = self._calculate_docs_hash(docs)

        # Controllo se esiste la cache valida per gli embedding vettoriali
        if not force_reload and os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                if cache_data.get("hash") == docs_hash and cache_data.get("embed_model") == self.embed_model:
                    self.chunks = [DocumentChunk.from_dict(c) for c in cache_data["chunks"]]
                    self.embeddings = np.array(cache_data["embeddings"], dtype=np.float32)
                    self.bm25.fit([c.text for c in self.chunks])
                    console.print(
                        f"[dim green][OK] Indice vettoriale e BM25 caricati ({len(self.chunks)} chunk semantici).[/dim green]"
                    )
                    return
            except Exception:
                pass

        # Generazione vettori densi con Ollama
        console.print(
            f"[cyan]Generazione embedding per {len(self.chunks)} chunk semantici tramite '{self.embed_model}'...[/cyan]"
        )
        embeddings_list = []
        failed_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("[green]Calcolo embeddings...", total=len(self.chunks))

            for chunk in self.chunks:
                emb = self._get_embedding(chunk.text)
                if emb is not None:
                    embeddings_list.append(emb)
                else:
                    failed_count += 1
                progress.advance(task)

        if failed_count > 0 or len(embeddings_list) != len(self.chunks):
            console.print("[yellow]Attenzione: Impossibile generare tutti gli embedding Ollama. Attivazione fallback TF-IDF.[/yellow]")
            self._setup_tfidf_fallback()
            return

        self.embeddings = np.array(embeddings_list, dtype=np.float32)
        # Normalizzazione per similarità del coseno
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.embeddings = self.embeddings / norms

        # Salvataggio in cache
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "hash": docs_hash,
                    "embed_model": self.embed_model,
                    "chunks": [c.to_dict() for c in self.chunks],
                    "embeddings": self.embeddings.tolist()
                }, f)
            console.print(f"[green][OK] Cache ibrida salvata ({len(self.chunks)} chunk).[/green]")
        except Exception as e:
            console.print(f"[dim yellow]Impossibile salvare la cache: {e}[/dim yellow]")

    def _setup_tfidf_fallback(self):
        """Configura il fallback TF-IDF se il modello Ollama per embeddings non è disponibile."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.use_tfidf_fallback = True
            corpus = [c.text for c in self.chunks]
            self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            console.print("[green][OK] Fallback TF-IDF configurato con successo.[/green]")
        except Exception as e:
            console.print(f"[bold red]Errore fallback TF-IDF: {e}[/bold red]")

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        dense_weight: float = 0.55,
        bm25_weight: float = 0.45,
        rrf_k: int = 60
    ) -> List[SearchResult]:
        """
        Esegue la Ricerca Ibrida combinando:
        1. Ricerca Semantica Densa (Cosine Similarity)
        2. Ricerca Lessicale BM25 (per codici, acronimi, numeri e date esatte)
        3. Reciprocal Rank Fusion (RRF) con boost per token tecnici esatti
        """
        n_chunks = len(self.chunks)
        if n_chunks == 0:
            return []

        # 1. Punteggi BM25
        bm25_scores = self.bm25.get_scores(query)
        if len(bm25_scores) == 0:
            bm25_scores = np.zeros(n_chunks, dtype=np.float32)

        # 2. Punteggi Densi
        dense_scores = np.zeros(n_chunks, dtype=np.float32)
        if self.use_tfidf_fallback:
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = self.tfidf_vectorizer.transform([query])
            dense_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        elif self.embeddings is not None:
            query_emb = self._get_embedding(query)
            if query_emb is not None:
                q_vec = np.array(query_emb, dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec = q_vec / q_norm
                dense_scores = np.dot(self.embeddings, q_vec)

        # 3. Boost per corrispondenze esatte di codici o date
        exact_boost = np.zeros(n_chunks, dtype=np.float32)
        query_tokens = BM25Index.tokenize(query)
        for t in query_tokens:
            is_code_or_num = any(char.isdigit() for char in t) or ("-" in t) or ("." in t) or len(t) > 6
            if is_code_or_num:
                for i, chunk in enumerate(self.chunks):
                    if t in chunk.text.lower():
                        exact_boost[i] += 0.25  # Forte bonus di posizionamento per codici/date esatte

        # 4. RRF (Reciprocal Rank Fusion)
        dense_rank_indices = np.argsort(dense_scores)[::-1]
        dense_ranks = np.zeros(n_chunks, dtype=np.float32)
        for rank, idx in enumerate(dense_rank_indices, 1):
            dense_ranks[idx] = rank

        bm25_rank_indices = np.argsort(bm25_scores)[::-1]
        bm25_ranks = np.zeros(n_chunks, dtype=np.float32)
        for rank, idx in enumerate(bm25_rank_indices, 1):
            bm25_ranks[idx] = rank

        rrf_scores = np.zeros(n_chunks, dtype=np.float32)
        for i in range(n_chunks):
            score_dense = dense_weight / (rrf_k + dense_ranks[i])
            score_bm25 = (bm25_weight / (rrf_k + bm25_ranks[i])) if bm25_scores[i] > 0 else 0.0
            rrf_scores[i] = score_dense + score_bm25 + exact_boost[i]

        max_bm25 = float(np.max(bm25_scores)) if np.max(bm25_scores) > 0 else 1.0
        norm_bm25 = bm25_scores / max_bm25
        norm_dense = np.clip(dense_scores, 0.0, 1.0)
        hybrid_scores = (dense_weight * norm_dense) + (bm25_weight * norm_bm25) + exact_boost

        top_indices = np.argsort(rrf_scores)[::-1][:top_k]

        results: List[SearchResult] = []
        for idx in top_indices:
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    hybrid_score=float(hybrid_scores[idx]),
                    dense_score=float(dense_scores[idx]),
                    bm25_score=float(bm25_scores[idx]),
                    rrf_score=float(rrf_scores[idx])
                )
            )

        return results

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Interfaccia compatibile con la versione PoC precedente."""
        hybrid_res = self.hybrid_search(query, top_k=top_k)
        return [(r.chunk, r.hybrid_score) for r in hybrid_res]


# =====================================================================
# 6. GESTIONE DELLA MEMORIA DI CONVERSAZIONE (CHAT MEMORY)
# =====================================================================

class ConversationTurn:
    """Rappresenta un singolo scambio di messaggi utente-assistente."""
    def __init__(self, query: str, response: str, sources: List[SearchResult]):
        self.query = query
        self.response = response
        self.sources = sources
        self.timestamp = time.time()


class ConversationMemory:
    """
    Gestisce la cronologia della conversazione per supportare follow-up questions
    e riferimenti a messaggi precedenti senza perdere il filo del discorso.
    """

    def __init__(self, max_turns: int = 6):
        self.turns: List[ConversationTurn] = []
        self.max_turns = max_turns

    def add_turn(self, query: str, response: str, sources: List[SearchResult]):
        self.turns.append(ConversationTurn(query=query, response=response, sources=sources))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_context_messages(self) -> List[Dict[str, str]]:
        """Restituisce la cronologia formattata per i messaggi chat di Ollama."""
        messages: List[Dict[str, str]] = []
        for t in self.turns:
            messages.append({"role": "user", "content": t.query})
            messages.append({"role": "assistant", "content": t.response})
        return messages

    def contextualize_query(self, query: str) -> str:
        """
        Riconosce se la domanda attuale è una follow-up question (es. 'Chi è?', 'Quali sono?')
        e la arricchisce con i riferimenti del turno precedente per ottimizzare il retrieval.
        """
        if not self.turns:
            return query

        anaphoric_signals = [
            "chi è", "cosa fa", "qual è", "quali sono", "e le sue", "e i suoi",
            "quanto costa", "quando", "dove", "questo", "questa", "questi", "loro",
            "invece", "inoltre", "perché", "riguardo", "dettagli"
        ]
        q_clean = query.strip().lower()
        is_followup = any(q_clean.startswith(sig) for sig in anaphoric_signals) or len(query.split()) <= 4

        if is_followup:
            last_turn = self.turns[-1]
            expanded = f"{query} [Riferimento al contesto precedente: {last_turn.query}]"
            return expanded

        return query

    def clear(self):
        self.turns.clear()


# =====================================================================
# 7. ARCHITETTURA MULTI-AGENTE: RICERCATORE E VALIDATORE
# =====================================================================

class ResearcherAgent:
    """
    AGENTE 1 (Ricercatore):
    - Analizza la query dell'utente tenendo conto della memoria storica
    - Esegue la Ricerca Ibrida (Dense + BM25) sul VectorStore
    - Estrae, ordina e formatta i contesti più rilevanti con etichette chiare delle fonti
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        memory: ConversationMemory,
        top_k: int = 3
    ) -> Tuple[List[SearchResult], str, str]:
        """Esegue il retrieval ibrido e formatta il contesto documentale."""
        search_query = memory.contextualize_query(query)
        results = self.vector_store.hybrid_search(search_query, top_k=top_k)

        if not results:
            context_text = "Nessun documento o frammento pertinente reperito nelle fonti locali."
        else:
            context_blocks = []
            for i, res in enumerate(results, 1):
                chunk = res.chunk
                source_name = Path(chunk.source).name
                page_info = f" (pag. {chunk.page})" if chunk.page else ""
                header = f"[Fonte {i}: {source_name}{page_info} | Rilevanza Ibrida: {res.hybrid_score:.2f} (Dense: {res.dense_score:.2f}, BM25: {res.bm25_score:.1f})]"
                context_blocks.append(f"{header}\n{chunk.text}")
            context_text = "\n\n---\n\n".join(context_blocks)

        return results, context_text, search_query


class ValidatorAgent:
    """
    AGENTE 2 (Validatore / Revisore):
    - Esamina la bozza generata da DeepSeek-R1 alla luce dei contesti documentali reali
    - Rileva e segnala allucinazioni o dichiarazioni prive di riscontro nelle fonti
    - Verifica che le citazioni [Fonte X] siano corrette e presenti
    - Rilascia un report di validazione fattuale e accuratezza
    """

    def __init__(self, ollama_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_LLM_MODEL):
        self.ollama_url = ollama_url
        self.model = model

    def validate_and_refine(
        self,
        draft_answer: str,
        retrieved_results: List[SearchResult],
        question: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Esegue la validazione euristica e di aderenza fattuale della risposta.
        Ritorna la risposta validata (o con avvisi di sicurezza) e il resoconto del controllo.
        """
        report = {
            "is_valid": True,
            "citations_found": [],
            "hallucinations_detected": 0,
            "verified_tokens": [],
            "status_message": "Verificata con successo",
            "confidence": 1.0
        }

        if not retrieved_results:
            report["status_message"] = "Nessuna fonte documentale disponibile per la verifica"
            report["confidence"] = 0.5
            return draft_answer, report

        # 1. Controllo Citazioni Fonti: cerca riferimenti [Fonte 1], [Fonte 2], ecc.
        citation_matches = re.findall(r"\[Fonte\s*(\d+)\]", draft_answer, re.IGNORECASE)
        available_sources_count = len(retrieved_results)
        valid_citations = []
        invalid_citations = []

        for cit in citation_matches:
            src_num = int(cit)
            if 1 <= src_num <= available_sources_count:
                valid_citations.append(src_num)
            else:
                invalid_citations.append(src_num)

        report["citations_found"] = sorted(list(set(valid_citations)))

        # 2. Controllo Preservazione Termini Critici (Codici, Date, Numeri)
        all_context_text = " ".join([r.chunk.text for r in retrieved_results]).lower()
        draft_tokens = BM25Index.tokenize(draft_answer)
        critical_tokens = [
            t for t in draft_tokens 
            if (any(c.isdigit() for c in t) or "-" in t or "." in t) and len(t) >= 2
        ]

        verified_tokens = []
        suspect_tokens = []
        for token in critical_tokens:
            if token in all_context_text:
                verified_tokens.append(token)
            else:
                suspect_tokens.append(token)

        report["verified_tokens"] = list(set(verified_tokens))

        # 3. Controllo Allucinazioni e Correzione
        if invalid_citations:
            report["hallucinations_detected"] += len(invalid_citations)
            for inv in invalid_citations:
                draft_answer = draft_answer.replace(f"[Fonte {inv}]", f"[Fonte non presente]")
            report["status_message"] = f"Rilevate citazioni inesistenti rimosse ({invalid_citations})"
            report["confidence"] -= 0.2

        if len(suspect_tokens) > 3:
            report["hallucinations_detected"] += 1
            report["status_message"] = "Possibile disallineamento su alcuni valori numerici o date rispetto ai documenti"
            report["confidence"] -= 0.15

        if available_sources_count > 0 and not report["citations_found"]:
            report["status_message"] = "Attenzione: la risposta non riportava citazioni esplicite alle fonti [Fonte X]"
            report["confidence"] -= 0.1

        return draft_answer, report


# =====================================================================
# 8. COORDINATORE RAG (RAG ASSISTANT)
# =====================================================================

class RAGAssistant:
    """Coordina l'intero ciclo RAG avanzato multi-agente, memoria e indicizzazione."""

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        llm_model: str = DEFAULT_LLM_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        directories: Optional[List[str]] = None
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.directories = directories or [".", "documenti", "docs"]

        # Componenti architetturali
        self.vector_store = VectorStore(ollama_url=self.ollama_url, embed_model=self.embed_model)
        self.memory = ConversationMemory(max_turns=6)
        self.researcher = ResearcherAgent(vector_store=self.vector_store)
        self.validator = ValidatorAgent(ollama_url=self.ollama_url, model=self.llm_model)

        self.last_retrieved_results: List[SearchResult] = []
        self.last_validation_report: Optional[Dict[str, Any]] = None

    def check_ollama(self) -> bool:
        """Verifica se Ollama è online e se il modello LLM è installato."""
        try:
            res = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if res.status_code == 200:
                data = res.json()
                models = [m.get("name") for m in data.get("models", [])]
                has_llm = any(self.llm_model in m for m in models)
                if not has_llm:
                    console.print(f"[bold red]Avviso: Il modello '{self.llm_model}' non è presente in Ollama.[/bold red]")
                    console.print(f"Modelli disponibili: {', '.join(models)}")
                    console.print(f"Puoi installarlo eseguendo: [bold cyan]ollama pull {self.llm_model}[/bold cyan]")
                return True
        except Exception:
            return False
        return False

    def load_and_index(self, force_reload: bool = False):
        """Scansiona e indicizza i documenti con hybrid search."""
        docs = DocumentLoader.scan_directory(self.directories)
        if not docs:
            console.print("[yellow]Nessun documento (.txt, .pdf, .md) trovato nelle cartelle configurate.[/yellow]")
            return
        self.vector_store.build(docs, force_reload=force_reload)

    def query(
        self,
        question: str,
        top_k: int = 3,
        show_thinking: bool = True,
        validate: bool = True
    ) -> str:
        """
        Esegue il flusso completo ad alta precisione a due agenti:
        1. Agente 1 (Ricercatore): Retrieval ibrido (Dense + BM25)
        2. Modello DeepSeek-R1: Generazione con catena di pensiero CoT
        3. Agente 2 (Validatore): Controllo anti-allucinazione e conformità citazioni
        """
        console.print(f"[bold magenta]🔍 Agente 1 (Ricercatore):[/bold magenta] Recupero contesti ibridi (Dense + BM25)...")
        results, context_text, expanded_query = self.researcher.retrieve(question, self.memory, top_k=top_k)
        self.last_retrieved_results = results

        if expanded_query != question:
            console.print(f"[dim]Contestualizzazione memoria: '{expanded_query}'[/dim]")

        # Prompt di sistema rigido per accuratezza fattuale
        system_prompt = (
            "Sei un assistente RAG aziendale di altissima precisione con supporto multi-agente. "
            "Rispondi in lingua italiana in modo chiaro, esaustivo e strutturato. "
            "DEVI BASARE LA TUA RISPOSTA ESCLUSIVAMENTE SUL CONTESTO DOCUMENTALE FORNITO. "
            "Non fare supposizioni e non inventare dati assenti dalle fonti. "
            "Se l'informazione non è reperibile nel testo fornito, dichiara espressamente: "
            "'I documenti forniti non contengono questa informazione.' "
            "Inserisci sempre riferimenti espliciti alle fonti per ogni fatto o dato riportato "
            "(es. [Fonte 1], [Fonte 2])."
        )

        history_msgs = self.memory.get_context_messages()
        current_user_content = (
            f"CONTESTO DOCUMENTALE ESTRATTO DAL RICERCATORE:\n"
            f"{context_text}\n\n"
            f"DOMANDA DELL'UTENTE:\n"
            f"{question}\n\n"
            f"RISPOSTA DOCUMENTATA:"
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_msgs)
        messages.append({"role": "user", "content": current_user_content})

        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.15,
                "top_p": 0.9
            }
        }

        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Errore nella connessione a Ollama: {e}[/bold red]")
            return ""

        in_thinking = False
        thinking_text = ""
        draft_answer = ""

        console.print()
        if show_thinking:
            console.print("[dim cyan]--- FASE DI RAGIONAMENTO (deepseek-r1 CoT) ---[/dim cyan]")

        buffer = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk_data = json.loads(line.decode("utf-8"))
                token = chunk_data.get("message", {}).get("content", "")
                if not token:
                    continue

                buffer += token

                # Gestione dei tag <think> ... </think> di DeepSeek-R1
                while buffer:
                    if not in_thinking:
                        if "<think>" in buffer:
                            idx = buffer.find("<think>")
                            before = buffer[:idx]
                            if before:
                                draft_answer += before
                            in_thinking = True
                            buffer = buffer[idx + 7:]
                        else:
                            if "<" in buffer and len(buffer) < 7:
                                break
                            draft_answer += buffer
                            buffer = ""
                    else:
                        if "</think>" in buffer:
                            idx = buffer.find("</think>")
                            think_part = buffer[:idx]
                            if show_thinking:
                                console.print(think_part, end="", style="dim italic cyan")
                            thinking_text += think_part
                            in_thinking = False
                            buffer = buffer[idx + 8:]
                            if show_thinking:
                                console.print("\n[dim cyan]-------------------------------------------[/dim cyan]\n")
                        else:
                            if "<" in buffer and len(buffer) < 8:
                                break
                            if show_thinking:
                                console.print(buffer, end="", style="dim italic cyan")
                            thinking_text += buffer
                            buffer = ""

            except Exception:
                continue

        if buffer:
            if in_thinking and show_thinking:
                console.print(buffer, end="", style="dim italic cyan")
            else:
                draft_answer += buffer

        draft_answer = draft_answer.strip()

        # FASE 2: Validatore / Revisore (Agente 2)
        if validate:
            console.print("[bold green]🛡️ Agente 2 (Validatore/Revisore):[/bold green] Verifica fedeltà fonti e anti-allucinazione...")
            validated_answer, report = self.validator.validate_and_refine(
                draft_answer=draft_answer,
                retrieved_results=results,
                question=question
            )
            self.last_validation_report = report
        else:
            validated_answer = draft_answer
            report = {"status_message": "Validazione disattivata", "confidence": 1.0}

        # Stampa della risposta finale validata
        console.print(f"\n[bold green]Risposta Finale Validata:[/bold green]")
        console.print(Markdown(validated_answer))

        # Stampa badge di validazione
        if validate and self.last_validation_report:
            status_style = "green" if self.last_validation_report.get("hallucinations_detected", 0) == 0 else "yellow"
            cit_info = f"Fonti citate: {self.last_validation_report.get('citations_found') or 'N/A'}"
            ver_info = f"Termini verificati: {len(self.last_validation_report.get('verified_tokens', []))}"
            msg = f"[{status_style}]Stato: {self.last_validation_report.get('status_message')} | {cit_info} | {ver_info}[/{status_style}]"
            console.print(Panel(msg, title="Certificato di Validazione Agente 2", border_style=status_style, expand=False))

        # Aggiorna la memoria di conversazione
        self.memory.add_turn(query=question, response=validated_answer, sources=results)

        console.print("\n")
        return validated_answer

    def print_sources_table(self):
        """Stampa una tabella dettagliata con metriche Ibride, Densa e BM25."""
        if not self.last_retrieved_results:
            console.print("[dim]Nessuna fonte documentale utilizzata.[/dim]")
            return

        table = Table(
            title="Fonti Documentali - Analisi Hybrid Search (Dense + BM25)",
            show_header=True,
            header_style="bold magenta"
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Documento", style="cyan", width=25)
        table.add_column("Pag.", style="yellow", width=5)
        table.add_column("Score Ibrido", style="bold green", width=12)
        table.add_column("Dense Cos.", style="dim blue", width=10)
        table.add_column("BM25", style="dim magenta", width=8)
        table.add_column("Anteprima Frammento", style="white")

        for i, res in enumerate(self.last_retrieved_results, 1):
            fname = Path(res.chunk.source).name
            pag = str(res.chunk.page) if res.chunk.page else "-"
            preview = res.chunk.text.replace("\n", " ")
            if len(preview) > 75:
                preview = preview[:72] + "..."
            table.add_row(
                str(i),
                fname,
                pag,
                f"{res.hybrid_score:.3f}",
                f"{res.dense_score:.3f}",
                f"{res.bm25_score:.1f}",
                preview
            )

        console.print(table)


# =====================================================================
# 9. INTERFACCIA INTERATTIVA CON MEMORIA
# =====================================================================

def run_interactive_shell(assistant: RAGAssistant, top_k: int = 3, show_thinking: bool = True):
    """Avvia la sessione interattiva con memoria, hybrid search e multi-agente."""
    console.print(Panel(
        f"[bold cyan]Sistema RAG Avanzato Multi-Agente & Hybrid Search[/bold cyan]\n"
        f"Modello LLM: [bold yellow]{assistant.llm_model}[/bold yellow] | "
        f"Embedding: [bold green]{assistant.embed_model}[/bold green] | "
        f"Motore Lessicale: [bold magenta]BM25Okapi[/bold magenta]\n"
        f"Agente 1: [cyan]Ricercatore Ibrido[/cyan] | Agente 2: [green]Validatore Anti-Allucinazione[/green]\n"
        f"Documenti indicizzati: [bold]{len(assistant.vector_store.chunks)} chunk semantici[/bold]\n\n"
        f"[dim]Comandi disponibili:[/dim]\n"
        f" * [bold white]fonti[/bold white]     : Dettagli su ranking ibrido, dense e BM25 usati per l'ultima risposta\n"
        f" * [bold white]memoria[/bold white]   : Visualizza i turni della conversazione corrente\n"
        f" * [bold white]pulisci[/bold white]   : Azzera la memoria della conversazione (nuovo argomento)\n"
        f" * [bold white]ricarica[/bold white]  : Riscansiona i documenti e ricostruisce gli indici vettoriali/BM25\n"
        f" * [bold white]think[/bold white]     : Attiva/disattiva il pensiero di DeepSeek-R1 (<think>)\n"
        f" * [bold white]esci[/bold white]      : Termina la sessione",
        title="RAG Multi-Agent Shell Interattiva",
        border_style="cyan"
    ))

    while True:
        try:
            console.print("[bold cyan]Domanda > [/bold cyan]", end="")
            question = input().strip()

            if not question:
                continue

            cmd = question.lower()
            if cmd in ["esci", "exit", "quit", "q"]:
                console.print("[yellow]Chiusura del sistema RAG. A presto![/yellow]")
                break
            elif cmd in ["fonti", "sources"]:
                assistant.print_sources_table()
                continue
            elif cmd in ["memoria", "storia", "history"]:
                if not assistant.memory.turns:
                    console.print("[dim yellow]La memoria della conversazione è attualmente vuota.[/dim yellow]")
                else:
                    console.print(f"[bold cyan]Cronologia Conversazione ({len(assistant.memory.turns)} turni):[/bold cyan]")
                    for idx, t in enumerate(assistant.memory.turns, 1):
                        console.print(f"[dim]#{idx}[/dim] [bold white]Utente:[/bold white] {t.query}")
                        short_resp = t.response.replace("\n", " ")
                        if len(short_resp) > 100:
                            short_resp = short_resp[:97] + "..."
                        console.print(f"    [dim green]Assistente:[/dim green] {short_resp}")
                continue
            elif cmd in ["pulisci", "reset", "clear_memory"]:
                assistant.memory.clear()
                console.print("[green][OK] Memoria della conversazione azzerata con successo.[/green]")
                continue
            elif cmd in ["ricarica", "reload"]:
                console.print("[cyan]Ricaricamento e reindicizzazione semantica e BM25 in corso...[/cyan]")
                assistant.load_and_index(force_reload=True)
                console.print(f"[green]Completato! {len(assistant.vector_store.chunks)} chunk semantici indicizzati.[/green]")
                continue
            elif cmd in ["think", "pensiero"]:
                show_thinking = not show_thinking
                status = "ATTIVATO" if show_thinking else "DISATTIVATO"
                console.print(f"[dim yellow]Visualizzazione ragionamento (thinking): {status}[/dim yellow]")
                continue
            elif cmd in ["aiuto", "help"]:
                console.print("[dim]Inserisci qualsiasi domanda oppure: 'fonti', 'memoria', 'pulisci', 'ricarica', 'think', 'esci'.[/dim]")
                continue
            elif cmd in ["clear", "cls"]:
                os.system("cls" if os.name == "nt" else "clear")
                continue

            assistant.query(question, top_k=top_k, show_thinking=show_thinking)
            assistant.print_sources_table()

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Interruzione rilevata. Chiusura...[/yellow]")
            break


# =====================================================================
# 10. MAIN ENTRYPOINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sistema RAG avanzato multi-agente con Hybrid Search (BM25 + Vettoriale) per documenti TXT e PDF."
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Domanda da porre direttamente ai documenti. Se omessa, avvia la shell interattiva."
    )
    parser.add_argument(
        "--docs-dir", "-d",
        nargs="+",
        default=[".", "documenti", "docs"],
        help="Directory o percorsi dei file da includere nell'analisi RAG."
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Numero di frammenti di contesto rilevanti da recuperare (default: 3)."
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_LLM_MODEL,
        help=f"Nome del modello Ollama LLM (default: {DEFAULT_LLM_MODEL})."
    )
    parser.add_argument(
        "--embed-model", "-e",
        default=DEFAULT_EMBED_MODEL,
        help=f"Nome del modello di embedding Ollama (default: {DEFAULT_EMBED_MODEL})."
    )
    parser.add_argument(
        "--reload", "-r",
        action="store_true",
        help="Forza la reindicizzazione e rigenerazione della cache degli embedding."
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Nasconde i passaggi di ragionamento intermedio (<think>) di DeepSeek-R1."
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disabilita il passaggio di verifica dell'Agente 2 (Validatore)."
    )

    args = parser.parse_args()

    # Inizializza l'assistente RAG
    assistant = RAGAssistant(
        llm_model=args.model,
        embed_model=args.embed_model,
        directories=args.docs_dir
    )

    # Verifica stato Ollama
    if not assistant.check_ollama():
        console.print("[bold red]Attenzione: Impossibile contattare Ollama su http://localhost:11434.[/bold red]")
        console.print("Assicurati che l'applicazione Ollama sia in esecuzione.")
        sys.exit(1)

    # Indicizzazione documenti
    assistant.load_and_index(force_reload=args.reload)

    # Esecuzione query singola o shell interattiva
    if args.query:
        console.print(f"[bold cyan]Domanda:[/bold cyan] {args.query}\n")
        assistant.query(
            args.query,
            top_k=args.top_k,
            show_thinking=not args.no_thinking,
            validate=not args.no_validation
        )
        assistant.print_sources_table()
    else:
        run_interactive_shell(assistant, top_k=args.top_k, show_thinking=not args.no_thinking)


if __name__ == "__main__":
    main()
