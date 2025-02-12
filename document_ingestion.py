from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import faiss

# 1. Ingest and chunk the document
pdf_path = "documents/constitution.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\nArticle"]
)
chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

# 2. Generate embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)

# 3. Persist the FAISS index and metadata (e.g., chunks) to disk
faiss.write_index(vector_store.index, "faiss_index.bin")
with open("chunks_metadata.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Precomputation complete. FAISS index and metadata saved.")