"""Test dei difetti corretti: ognuno falliva sulla versione precedente del codice."""
import sys
from pathlib import Path

import pytest
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rag_demo as r


def store(testi):
    vs = r.VectorStore()
    vs.chunks = [r.DocumentChunk(t, f"doc{i}.txt", chunk_id=i) for i, t in enumerate(testi)]
    vs.bm25.fit(testi)
    return vs


def test_tokenizer_conserva_le_lettere_accentate():
    assert r.BM25Index.tokenize("perché la città SN-900") == ["perché", "la", "città", "sn-900"]


def test_abbreviazioni_e_versioni_non_spezzano_la_frase():
    frasi = r.SemanticTextSplitter._split_into_sentences("Vedi es. Modello X e pag. 12. Poi la v1. Fine.")
    assert frasi == ["Vedi es. Modello X e pag. 12.", "Poi la v1. Fine."]


def test_il_bonus_non_domina_il_ranking():
    vs = store(["Il sensore misura la temperatura", "Testo sul progetto precedente"])
    assert max(x.rrf_score for x in vs.hybrid_search("progetto precedente", top_k=2)) < 0.1


def test_la_memoria_non_premia_parole_di_servizio():
    vs = store([
        "Il sensore SN-900 misura la temperatura.",
        "Riferimento al contesto precedente della riunione sul progetto.",
    ] + [f"Argomento diverso numero {i}" for i in range(5)])
    mem = r.ConversationMemory()
    mem.add_turn("Quali sensori ci sono?", "risposta", [])
    risultati, _, _ = r.ResearcherAgent(vs).retrieve("Dettagli SN-900?", mem, top_k=1)
    assert "SN-900" in risultati[0].chunk.text


def test_codice_cercato_per_token_intero():
    vs = store(["Modello SN-9000 obsoleto", "Modello SN-900 attuale"])
    assert "SN-900 " in vs.hybrid_search("SN-900", top_k=1)[0].chunk.text


def test_validatore_segnala_numeri_inventati():
    chunk = r.DocumentChunk("Il sensore SN-900 dura 36 mesi.", "a.txt")
    risposta, report = r.ValidatorAgent().validate_and_refine(
        "Il sensore costa 5000 euro [Fonte 1].", [r.SearchResult(chunk, 1.0)], "q"
    )
    assert report["is_valid"] is False
    assert report["suspect_tokens"] == ["5000"]


def test_validatore_non_confonde_numeri_parziali():
    chunk = r.DocumentChunk("Fondato nel 1999.", "a.txt")
    _, report = r.ValidatorAgent().validate_and_refine("Dura 99 anni [Fonte 1].", [r.SearchResult(chunk, 1.0)], "q")
    assert report["suspect_tokens"] == ["99"]


def test_percorso_relativo_e_readme_escluso(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("ciao", encoding="utf-8")
    (tmp_path / "README.md").write_text("non indicizzare", encoding="utf-8")
    monkeypatch.chdir(docs)
    assert len(r.DocumentLoader.scan_directory(["../docs"])) == 1
    monkeypatch.chdir(tmp_path)
    assert list(r.DocumentLoader.scan_directory(r.DEFAULT_DOC_DIRS)) == [str(Path("docs/a.txt"))]


def test_cartella_nascosta_ignorata_dentro_i_documenti(tmp_path):
    (tmp_path / ".privata").mkdir()
    (tmp_path / ".privata" / "x.txt").write_text("no", encoding="utf-8")
    assert r.DocumentLoader.scan_directory([str(tmp_path)]) == {}


@pytest.mark.parametrize("modello,atteso", [("nomic-embed-text", "search_query: ciao"), ("altro", "ciao")])
def test_prefisso_embedding(monkeypatch, modello, atteso):
    inviati = []

    class Risposta:
        status_code = 200

        def json(self):
            return {"embeddings": [[1.0]]}

    def finto_post(url, json, timeout):
        inviati.append(json["input"])
        return Risposta()

    monkeypatch.setattr(r.requests, "post", finto_post)
    r.VectorStore(embed_model=modello)._get_embedding("ciao", task="search_query")
    assert inviati == [atteso]


def test_testo_con_parentesi_non_rompe_la_tabella_fonti():
    assistant = r.RAGAssistant()
    chunk = r.DocumentChunk("oggetto [/Type] pdf", "a.txt")
    assistant.last_retrieved_results = [r.SearchResult(chunk, 1.0)]
    vecchia = r.console
    r.console = Console(file=open("/dev/null" if sys.platform != "win32" else "NUL", "w"))
    try:
        assistant.print_sources_table()
    finally:
        r.console = vecchia


def test_flusso_completo_con_ollama_simulato(monkeypatch):
    righe = [
        b'{"message":{"content":"<think>rifletto [/x]"}}',
        b'{"message":{"content":"</think>"}}',
        b'{"message":{"content":"La batteria dura 36 mesi [Fonte 1]."}}',
    ]

    class Risposta:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(righe)

    monkeypatch.setattr(r.requests, "post", lambda *a, **k: Risposta())
    assistant = r.RAGAssistant()
    testo = "La batteria dura 36 mesi."
    assistant.vector_store.chunks = [r.DocumentChunk(testo, "a.txt")]
    assistant.vector_store.bm25.fit([testo])
    risposta = assistant.query("Quanto dura la batteria?")
    assert risposta == "La batteria dura 36 mesi [Fonte 1]."
    assert assistant.last_validation_report["is_valid"] is True
    assert len(assistant.memory.turns) == 1
