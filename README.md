
# Chat-with-Code (Local AI Agents)

Ask questions about any codebase and get answers grounded in retrieved source code context, using a local Ollama model + LangChain Retrieval-Augmented Generation (RAG).

## What this project does

This repo is a tiny “chat with a codebase” CLI:

1. Indexes a target folder by reading source files, splitting them into chunks, embedding those chunks, and storing them in a local Chroma vector database.
2. Runs an interactive loop where each question retrieves the most relevant chunks and injects them into the prompt sent to a local Ollama LLM.

## How it works (RAG pipeline)

For each question:

`question` → embed the question → vector similarity search (Chroma) → top-`k` chunks → prompt construction → Ollama LLM answer

## Project structure

- `main.py`: Interactive Q&A loop. Builds the prompt and calls the Ollama LLM.
- `codebase.py`: Indexing + retrieval. Walks a codebase, chunks files, embeds chunks, writes to Chroma, exposes `retriever`.
- `chroma_code_db/`: Local persisted Chroma database (created/updated when indexing runs).
- `Guide.md`: Long-form, line-by-line walkthrough of the code.

## Run

The program prompts you for a codebase path (because `main.py` imports `retriever` from `codebase.py`, which executes indexing at import time).

```powershell
uv run main.py
```

If you’re not using `uv`, you can also run with your Python launcher:

```powershell
python main.py
```

Then:

1. Enter the path to the codebase you want to index.
2. Ask questions in the terminal.
3. Type `q` to quit.

## Key settings you can tune

All tuning knobs live in `codebase.py`.

### Retrieval depth (`k`)

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 10})
```

- Increase `k` (e.g. 12–16) if answers miss cross-file context.
- Decrease `k` (e.g. 4–8) if answers include lots of irrelevant code.

### Chunking (`chunk_size`, `chunk_overlap`)

```python
splitter = RecursiveCharacterTextSplitter(
	chunk_size=800,
	chunk_overlap=100
)
```

These are **character** counts (not tokens).

- Larger `chunk_size` keeps more code together (better for “explain the flow”).
- Smaller `chunk_size` makes retrieval more precise (better for “where is X defined?”).
- `chunk_overlap` prevents important lines from being split across chunk boundaries.

Practical starting point for code RAG:

- `chunk_size`: 1200–2000
- `chunk_overlap`: 150–300

### Models

- In `main.py` (generation): `OllamaLLM(model="llama3.2")`
- In `codebase.py` (embeddings): `OllamaEmbeddings(model="mxbai-embed-large")`

If you change these strings, you are changing which Ollama models are used.

## Notes / limitations

- Re-running indexing can add duplicate chunks to the same Chroma collection unless you add explicit de-duplication or collection resets.
- Only these file extensions are indexed by default: `.py`, `.ts`, `.js`, `.tsx`.

## Troubleshooting (common)

## 🌟 Support the Project
If this project helped you understand RAG or helped you build your own local AI tool:
- **Star** this repository to show your support!
- **Fork** it to experiment with different LLMs or data loaders.
- **Open an Issue** if you find a bug or have a feature request.

- **Empty/low-quality answers**: increase `k`, increase `chunk_size`, or ensure the indexed folder is the right project root.
- **Model not found**: make sure the named Ollama models exist locally (the strings in `main.py` / `codebase.py` must match what Ollama has).
- **Retrieval feels noisy**: reduce `k`, reduce `chunk_size`, or avoid indexing generated/vendor folders in the target codebase.

