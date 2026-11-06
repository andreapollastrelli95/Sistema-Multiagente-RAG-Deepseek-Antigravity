"""Misura hit rate e MRR sui documenti demo.

Senza --ollama usa TF-IDF + BM25 e non richiede modelli. Un risultato conta come
trovato se il chunk viene dal file atteso e contiene il testo atteso.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rag_demo as r


def costruisci_indice(cartella, usa_ollama):
    documenti = r.DocumentLoader.scan_directory([str(cartella)])
    store = r.VectorStore()
    if usa_ollama:
        store.build(documenti, force_reload=True)
        return store
    splitter = r.SemanticTextSplitter()
    store.chunks = [
        r.DocumentChunk(testo, percorso, pagina, chunk_id=i)
        for i, (percorso, testo, pagina) in enumerate(
            (p, t, pag)
            for p, contenuti in documenti.items()
            for contenuto, pag in contenuti
            for t in splitter.split_text(contenuto)
        )
    ]
    store.bm25.fit([c.text for c in store.chunks])
    store._setup_tfidf_fallback()
    return store


def valuta(store, domande, k):
    trovate, reciproci, errori = 0, 0.0, []
    for d in domande:
        risultati = store.hybrid_search(d["domanda"], top_k=k)
        rango = next(
            (i for i, x in enumerate(risultati, 1)
             if Path(x.chunk.source).name == d["fonte"] and d["testo"].lower() in x.chunk.text.lower()),
            None,
        )
        if rango:
            trovate += 1
            reciproci += 1 / rango
        else:
            errori.append(d["domanda"])
    return {"domande": len(domande), "k": k, "hit_rate": trovate / len(domande),
            "mrr": reciproci / len(domande), "non_trovate": errori}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--ollama", action="store_true", help="Usa gli embedding di Ollama")
    args = parser.parse_args()
    domande = json.loads((ROOT / "evaluation" / "domande.json").read_text(encoding="utf-8"))
    print(json.dumps(valuta(costruisci_indice(ROOT / "documenti", args.ollama), domande, args.k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
