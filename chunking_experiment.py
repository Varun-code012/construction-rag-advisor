from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import shutil
import os

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Setup
PDF = "sample2.pdf"
QUESTION = "What are the challenges in AI?"

# Load PDF once
loader = PyPDFLoader(PDF)
documents = loader.load()

# Embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=API_KEY)

# Test 3 chunk sizes
for chunk_size in [200, 500, 1000]:
    print(f"\n{'='*50}")
    print(f"CHUNK SIZE: {chunk_size}")
    print(f"{'='*50}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Sample chunk:\n{chunks[0].page_content[:200]}")

    db_path = f"./chroma_test_{chunk_size}"
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke(QUESTION)
    context = "\n\n".join([doc.page_content for doc in docs])
    response = llm.invoke(
        f"Answer this question based on the context below.\n\nContext:\n{context}\n\nQuestion: {QUESTION}"
    )

    print(f"\nQuestion: {QUESTION}")
    print(f"Answer: {response.content}")

    vectorstore._client._system.stop()
    del vectorstore
    shutil.rmtree(db_path, ignore_errors=True)

print("\nExperiment complete!")