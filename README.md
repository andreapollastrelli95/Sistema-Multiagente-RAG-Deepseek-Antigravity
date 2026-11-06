# RAG locale con ricerca ibrida

Interroga documenti `.txt`, `.md` e `.pdf` in linguaggio naturale, in italiano, con modelli eseguiti sul proprio computer tramite [Ollama](https://ollama.com). Documenti e domande restano in locale.

Ogni risposta indica le fonti usate e passa da un controllo automatico di citazioni, numeri e codici.

## Come funziona

1. **Chunking:** divide i testi rispettando titoli, paragrafi e frasi.
2. **Ricerca ibrida:** unisce con Reciprocal Rank Fusion la similarità degli embedding e BM25, con un piccolo bonus per i codici esatti come `SN-900`.
3. **Risposta:** `deepseek-r1:8b` risponde usando solo i frammenti recuperati e li cita come `[Fonte 1]`.
4. **Controllo euristico:** segnala citazioni inesistenti e numeri, date o codici assenti dalle fonti.

Il controllo non usa un LLM e non comprende il significato delle frasi: un esito positivo non dimostra che la risposta sia corretta.

Se gli embedding non sono disponibili durante l'indicizzazione, il sistema passa a TF-IDF. Per generare la risposta Ollama resta necessario.

## Avvio

```bash
pip install -r requirements.txt
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text

python rag_demo.py                                          # shell interattiva
python rag_demo.py "Chi è il coordinatore del Progetto Aurora?"   # domanda singola
```

La cartella `documenti/` contiene due file dimostrativi con dati fittizi. Per usare i propri: `python rag_demo.py --docs-dir percorso/cartella`.

| Opzione | Effetto | Predefinito |
|---|---|---|
| `--docs-dir`, `-d` | Cartelle o file da indicizzare | `documenti`, `docs` |
| `--top-k`, `-k` | Frammenti passati al modello | `3` |
| `--reload`, `-r` | Rigenera la cache degli embedding | disattivo |
| `--no-thinking` | Nasconde il ragionamento del modello | disattivo |
| `--no-validation` | Disattiva il controllo | disattivo |
| `--model`, `-m` | Modello LLM | `deepseek-r1:8b` |
| `--embed-model`, `-e` | Modello di embedding | `nomic-embed-text` |

Comandi nella shell: `fonti`, `memoria`, `pulisci`, `ricarica`, `think`, `esci`.

## Test e valutazione

```bash
pip install -r requirements-dev.txt
pytest
python evaluation/valuta_retrieval.py
```

Sulle 12 domande di `evaluation/domande.json`, in modalità TF-IDF + BM25 senza Ollama, il frammento corretto è tra i primi 3 in tutti i casi (hit rate 1,00, MRR 0,96).

Il corpus ha solo due documenti e 7 frammenti: la misura serve a intercettare regressioni, non a stimare la qualità su documenti reali. Con Ollama attivo si può ripetere usando `--ollama`.

## Limiti

- La qualità della risposta non è valutata: la misura riguarda solo il recupero dei frammenti.
- I PDF scansionati non sono supportati perché manca l'OCR.
- Gli indici stanno in memoria e la cache è un file JSON: la soluzione è adatta a piccole raccolte.
