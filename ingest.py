from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("Loading BBMP PDF...")
loader = PyPDFLoader("bbmp.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

print("Creating embeddings and storing in ChromaDB...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True}
)
vectorstore = Chroma.from_documents(
    chunks, embeddings, persist_directory="./chroma_db"
)
print("Done! ChromaDB is ready.")