"""
test_rag_advanced.py - Test automatici per l'architettura RAG Avanzata Multi-Agente
Verifica:
1. Hybrid Search & BM25 (Codici tecnici, date, termini esatti)
2. Semantic Chunking (Sezioni logiche, titoli e paragrafi)
3. Agenti Ricercatore e Validatore (Controllo anti-allucinazione e citazioni)
4. Chat Memory (Cronologia e follow-up context)
5. Ingestion e Retrieval reale da directory documenti
"""

import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_demo import (
    BM25Index,
    SemanticTextSplitter,
    DocumentChunk,
    SearchResult,
    VectorStore,
    ResearcherAgent,
    ValidatorAgent,
    ConversationMemory,
    DocumentLoader,
)


class TestBM25Index(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "Specifiche Hardware: I sensori ambientali modello SN-900 operano tra -20C e +60C.",
            "La batteria al litio integrata garantisce 36 mesi di autonomia continua.",
            "Progetto Aurora: coordinatore tecnico Marco Rossi. Aggiornamenti rilasciati ogni due settimane.",
            "Codice versione v1.2.0 rilasciata a settembre 2026 con crittografia AES-256."
        ]
        self.bm25 = BM25Index()
        self.bm25.fit(self.corpus)

    def test_tokenization_codes_and_dates(self):
        tokens = BM25Index.tokenize("Sensore SN-900 v1.2.0 data 2026-09-13")
        self.assertIn("sn-900", tokens)
        self.assertIn("v1.2.0", tokens)
        self.assertIn("2026-09-13", tokens)

    def test_keyword_search_precision(self):
        # Ricerca per codice esatto hardware
        scores = self.bm25.get_scores("SN-900")
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])

        # Ricerca per data e versione
        scores_version = self.bm25.get_scores("v1.2.0 settembre 2026")
        self.assertGreater(scores_version[3], scores_version[0])

        # Ricerca per persona
        scores_person = self.bm25.get_scores("Marco Rossi")
        self.assertGreater(scores_person[2], scores_person[0])


class TestSemanticTextSplitter(unittest.TestCase):
    def test_section_and_paragraph_boundaries(self):
        doc = (
            "# Architettura Sistema\n\n"
            "Il sistema Aurora utilizza un approccio RAG locale ad alta sicurezza.\n"
            "Tutti i dati rimangono sulla macchina dell'utente.\n\n"
            "## Parametri Operativi\n\n"
            "La porta Ollama predefinita è 11434. Modello deepseek-r1:8b.\n"
            "La temperatura consigliata è 0.2 per compiti di estrazione fattuale."
        )
        splitter = SemanticTextSplitter(target_chunk_size=180, min_chunk_size=40)
        chunks = splitter.split_text(doc)

        self.assertGreaterEqual(len(chunks), 2)
        # Verifica che non abbia tagliato parole o spezzato frasi a metà
        for c in chunks:
            self.assertTrue(c.strip())
            self.assertFalse(c.endswith("deepseek-"))
            self.assertFalse(c.endswith("approccio"))


class TestMultiAgentArchitecture(unittest.TestCase):
    def setUp(self):
        self.store = VectorStore()
        # Mock chunks
        self.store.chunks = [
            DocumentChunk(
                text="I sensori ambientali SN-900 operano tra -20C e +60C con batteria di 36 mesi.",
                source="scheda_dispositivi.pdf",
                page=1,
                chunk_id=0
            ),
            DocumentChunk(
                text="Il coordinatore del Progetto Aurora è Marco Rossi. Rilasciata versione v1.2.0 a settembre 2026.",
                source="guida_progetto.txt",
                page=None,
                chunk_id=1
            )
        ]
        self.store.bm25.fit([c.text for c in self.store.chunks])
        self.researcher = ResearcherAgent(vector_store=self.store)
        self.validator = ValidatorAgent()
        self.memory = ConversationMemory()

    def test_researcher_hybrid_retrieval(self):
        results, context_text, query = self.researcher.retrieve("SN-900 batteria", self.memory, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("scheda_dispositivi.pdf", results[0].chunk.source)
        self.assertIn("[Fonte 1:", context_text)

    def test_validator_hallucination_and_citations(self):
        results, _, _ = self.researcher.retrieve("SN-900", self.memory, top_k=1)

        # Risposta fedele con citazione corretta
        faithful_answer = "I sensori SN-900 operano tra -20C e +60C [Fonte 1]."
        _, report1 = self.validator.validate_and_refine(faithful_answer, results, "Specifiche?")
        self.assertTrue(report1["is_valid"])
        self.assertEqual(report1["suspect_tokens"], [])
        self.assertIn(1, report1["citations_found"])

        # Risposta con citazione inesistente [Fonte 5]
        hallucinated_cit = "I sensori SN-900 hanno schermo OLED [Fonte 5]."
        corrected_answer, report2 = self.validator.validate_and_refine(hallucinated_cit, results, "Specifiche?")
        self.assertFalse(report2["is_valid"])
        self.assertEqual(report2["invalid_citations"], [5])
        self.assertIn("[Fonte non presente]", corrected_answer)


class TestChatMemory(unittest.TestCase):
    def test_followup_contextualization(self):
        mem = ConversationMemory()
        dummy_chunk = DocumentChunk(text="Marco Rossi è il coordinatore", source="doc.txt")
        sr = SearchResult(chunk=dummy_chunk, hybrid_score=0.9)
        mem.add_turn("Chi è il coordinatore di Aurora?", "È Marco Rossi [Fonte 1].", [sr])

        # Follow up anaphorico
        q2 = "Quali sono i suoi compiti?"
        contextualized = mem.contextualize_query(q2)
        self.assertIn("Chi è il coordinatore di Aurora?", contextualized)

        # Reset memoria
        mem.clear()
        self.assertEqual(len(mem.turns), 0)


class TestRealDocumentIngestion(unittest.TestCase):
    def test_scan_and_hybrid_search(self):
        docs_dir = Path("documenti")
        if not docs_dir.exists():
            self.skipTest("Cartella documenti non trovata.")

        docs = DocumentLoader.scan_directory(["documenti"])
        self.assertGreater(len(docs), 0)

        store = VectorStore()
        # Build index con TF-IDF / BM25
        store.use_tfidf_fallback = True
        splitter = SemanticTextSplitter()
        chunks = []
        for fpath, contents in docs.items():
            for text, page in contents:
                for c in splitter.split_text(text):
                    chunks.append(DocumentChunk(text=c, source=fpath, page=page, chunk_id=len(chunks)))
        store.chunks = chunks
        store.bm25.fit([c.text for c in chunks])
        store._setup_tfidf_fallback()

        # Cerca codice esatto del PDF
        res_pdf = store.hybrid_search("SN-900", top_k=1)
        self.assertGreater(len(res_pdf), 0)
        self.assertIn("scheda_dispositivi.pdf", res_pdf[0].chunk.source)

        # Cerca persona dal file TXT
        res_txt = store.hybrid_search("Marco Rossi", top_k=1)
        self.assertGreater(len(res_txt), 0)
        self.assertIn("guida_progetto.txt", res_txt[0].chunk.source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
