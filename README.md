# Construction & Building Code Advisor
A RAG-based system that answers questions about BBMP Building Bye-Laws 2003.

## Tech Stack
- LangChain — orchestration
- ChromaDB — vector database
- Groq (llama-3.3-70b-versatile) — LLM
- sentence-transformers (all-MiniLM-L6-v2) — embeddings
- Streamlit — frontend (coming soon)

## Setup
1. Clone the repo
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install packages: `pip install -r requirements.txt`
5. Add your Groq API key to `.env` file
6. Run ingestion: `python ingest_cleaned.py`
7. Query: `python query.py`

## Project Structure
- `ingest_cleaned.py` — loads and indexes BBMP PDF into ChromaDB
- `query.py` — queries the RAG pipeline
- `clean_pdf.py` — cleans raw BBMP PDF text
- `bbmp.pdf` — BBMP Building Bye-Laws 2003 (not in repo, download separately)

## Pipeline Settings
- chunk_size: 1500
- chunk_overlap: 150
- k: 6
- temperature: 0