# Sistema RAG con Modelli Locali Ollama (deepseek-r1:8b)

Un sistema di **Retrieval-Augmented Generation (RAG)** completo, veloce e sicuro per analizzare ed effettuare domande in linguaggio naturale sui tuoi documenti locali (`.txt`, `.pdf`, `.md`).

I dati rimangono al 100% sulla tua macchina:
- **LLM locale**: `deepseek-r1:8b` (in esecuzione su `http://localhost:11434`)
- **Embeddings semantici**: `nomic-embed-text` (Ollama)
- **Motore di ricerca**: Ricerca semantica vettoriale su coseno con fallback integrato TF-IDF (`scikit-learn`)
- **Parser PDF**: `pypdf` ad alte prestazioni
- **Interfaccia CLI**: Basata su `rich` con supporto a streaming in tempo reale, tabelle delle fonti ed evidenziazione della fase di ragionamento (*thinking* di DeepSeek).

---

## 📁 Struttura del Progetto

```
Progetti Ai/
│
├── rag_demo.py                  # Script principale del sistema RAG
├── .rag_cache.json              # Cache automatica degli embedding vettoriali (generata al primo avvio)
├── README.md                    # Questa guida
│
└── documenti/                   # Cartella per i tuoi file da analizzare
    ├── guida_progetto.txt       # Esempio testuale (.txt)
    └── scheda_dispositivi.pdf   # Esempio PDF (.pdf)
```

---

## 🚀 Prerequisiti e Avvio

1. Assicurati che **Ollama** sia in esecuzione in background:
   ```bash
   ollama list
   ```
   Dovresti visualizzare `deepseek-r1:8b` e `nomic-embed-text`. Se necessario, puoi scaricarli con:
   ```bash
   ollama pull deepseek-r1:8b
   ollama pull nomic-embed-text
   ```

2. Le dipendenze Python necessarie (`pypdf`, `rich`, `requests`, `scikit-learn`, `numpy`, `langchain-text-splitters`) sono già installate.

---

## 💡 Modalità d'Uso

### 1. Modalità Interattiva (Shell nel Terminale)
Avvia la sessione interattiva di domande e risposte:
```bash
python rag_demo.py
```

All'interno della shell potrai digitare le tue domande oppure utilizzare i seguenti comandi:
- `fonti`: Mostra una tabella dettagliata con i frammenti documentali usati nell'ultima risposta (file, pagina, punteggio ibrido, punteggio denso e BM25, anteprima).
- `memoria`: Visualizza i turni recenti della conversazione corrente (utile per monitorare il contesto).
- `pulisci`: Azzera la cronologia della conversazione per avviare una nuova sessione da zero.
- `ricarica`: Riscansiona la cartella documenti e rigenera l'indice vettoriale e BM25 (utile dopo modifiche ai file).
- `think`: Attiva o disattiva la visualizzazione a schermo dei passaggi di ragionamento (`<think>`) del modello DeepSeek.
- `esci` o `q`: Termina la sessione.

---

### 2. Modalità a Singola Domanda (One-Shot)
Puoi passare una domanda direttamente come argomento da riga di comando:
```bash
python rag_demo.py "Quali sono le specifiche dei sensori ambientali SN-900?"
```

```bash
python rag_demo.py "Chi è il coordinatore tecnico del Progetto Aurora e ogni quanto rilasciano aggiornamenti?"
```

---

## ⚙️ Opzioni da Riga di Comando

| Argomento | Descrizione | Default |
|---|---|---|
| `query` | Domanda opzionale. Se omessa, avvia la shell interattiva. | `None` |
| `--docs-dir`, `-d` | Cartelle o singoli file da analizzare. | `.` e `documenti` |
| `--top-k`, `-k` | Numero di frammenti più rilevanti da passare al modello. | `3` |
| `--reload`, `-r` | Forza la rigenerazione della cache degli embedding. | `False` |
| `--no-thinking` | Nasconde a terminale i passaggi intermedi `<think>`. | `False` |
| `--no-validation` | Disabilita il passaggio di verifica dell'Agente 2 (Validatore). | `False` |
| `--model`, `-m` | Nome del modello LLM su Ollama. | `deepseek-r1:8b` |
| `--embed-model`, `-e` | Nome del modello di embedding su Ollama. | `nomic-embed-text` |

---

## ⚡ Caratteristiche Chiave Avanzate

1. **Hybrid Search & RRF Reranking**:
   Combina la ricerca semantica vettoriale (cosine similarity) con la ricerca lessicale BM25Okapi e un boost per token tecnici esatti. Impedisce la perdita di codici hardware (es. `SN-900`), versioni software (`v1.2.0`), date e codici alfanumerici.
2. **Chunking Semantico Strutturato**:
   Segmenta i documenti rispettando intestazioni Markdown (`#`, `##`), sezioni numerate, doppi ritorni a capo (paragrafi) e confini di frase completi, evitando frammentazioni artificiali a metà frase.
3. **Architettura Multi-Agente (Ricercatore + Validatore/Revisore)**:
   - **Agente 1 (Ricercatore)**: Esegue il retrieval ibrido, contestualizza la richiesta e formula i contesti documentali strutturati.
   - **Agente 2 (Validatore/Revisore)**: Ispeziona la risposta di DeepSeek-R1 prima dell'invio all'utente, verifica l'aderenza fattuale alle fonti fornite, convalida le citazioni esplicite `[Fonte X]` e segnala o elimina eventuali allucinazioni.
4. **Chat Memory (Memoria di Conversazione)**:
   Mantiene una finestra scorrevole dei turni di dialogo precedenti, consentendo follow-up questions fluide con risoluzione automatica delle domande con riferimenti impliciti.
5. **Caching Intelligente degli Embedding (`.rag_cache.json`)**:
   Calcola un hash MD5 dello stato dei file. Se i documenti non sono stati modificati, l'avvio e l'indicizzazione sono istantanei.
6. **Fallback Resiliente**:
   Se per qualsiasi motivo il modello di embedding locale o Ollama non risponde, il sistema attiva automaticamente il calcolo TF-IDF / BM25 senza bloccarsi.
