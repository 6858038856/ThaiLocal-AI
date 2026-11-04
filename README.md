# ThaiLocal AI — Local Knowledge RAG Assistant

A course demonstration of a Thai-food knowledge assistant. It uses locally running Ollama, Qwen3 1.7B, Nomic embedding model, and user-uploaded PDF/TXT/MD files. Answers show source filenames/page numbers, and uploaded documents can be downloaded by clicking their filenames.

## Setup (on a laptop or desktop)

1. Install Python 3.10+ and [Ollama](https://ollama.com/download).
2. Pull models: `ollama pull qwen3:1.7b` and `ollama pull nomic-embed-text`.
3. Start Ollama if not already running: `ollama serve`.
4. Install Python packages: `pip install -r requirements.txt`.
5. Run app: `streamlit run app.py`.
6. Open Streamlit in your browser; in **Document collection**, upload 4–5 Thai-food PDF files and click **Index selected files**.
7. Ask questions in the chat. Check the filename and page citations, and click documents in the sidebar to download them.

## How it works

- PDF text is extracted with `pypdf` and split into overlapping chunks.
- Ollama's `nomic-embed-text` creates embeddings for document chunks and the user's query.
- Cosine similarity retrieves the closest chunks.
- `qwen3:1.7b` responds with the retrieved evidence and is asked to cite its sources.

## Important limitations

- Requires Ollama running locally with both models; this app is not hosted on GitHub.
- The current collection is stored in Streamlit session state and is cleared when the session ends; re-upload after restart.
- Scanned/image-only PDFs need OCR, which is not included.
- Source passages are retrieved by similarity; human verification is still necessary.
- **Do not put third-party PDFs in the public GitHub repository.** Upload them only through your own local app.

## Team

- Nichakarn Leenabanchong (Nini), 6858038856
- Kotchaphon Neranartkomol (Ferri), 6858001556
