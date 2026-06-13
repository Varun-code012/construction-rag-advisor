from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import shutil
from dotenv import load_dotenv
import os

# Config
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Test queries — real BBMP questions
PDF = "bbmp.pdf"
QUESTIONS = [
    "What are the set-back requirements in Table 4 for residential buildings?",
    "What is the maximum height allowed for residential buildings?",
    "What documents are required for building plan approval?"
]

# Load PDF once
print("Loading BBMP PDF...")
loader = PyPDFLoader(PDF)
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# Embeddings and LLM — load once
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=API_KEY)

# Combinations to test
CHUNK_SIZES = [500, 1000, 1500]
K_VALUES = [4, 6]

results = []

for chunk_size in CHUNK_SIZES:
    for k in K_VALUES:
        print(f"\n{'='*55}")
        print(f"Testing: chunk_size={chunk_size}, k={k}")
        print(f"{'='*55}")

        # Chunk the documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1)
        )
        chunks = splitter.split_documents(documents)
        print(f"Total chunks: {len(chunks)}")

        # Store in ChromaDB
        db_path = f"./chroma_tune_{chunk_size}_{k}"
        vectorstore = Chroma.from_documents(
            chunks, embeddings, persist_directory=db_path
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        # Test all 3 questions
        for i, question in enumerate(QUESTIONS):
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            response = llm.invoke(
                f"Answer this question based only on the context below. "
                f"If the answer is not in the context, say 'Not found'.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
            print(f"\nQ{i+1}: {question}")
            print(f"A: {response.content[:300]}...")

        # Cleanup
        vectorstore._client._system.stop()
        del vectorstore
        shutil.rmtree(db_path, ignore_errors=True)

print("\n\nExperiment complete! Pick the best combination.")