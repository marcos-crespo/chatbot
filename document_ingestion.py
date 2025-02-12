from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Ingest and chunk the document
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

# Build the vector store using embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)

# Save the vector store locally (this saves index, docstore, and index_to_docstore_id)
vector_store.save_local("faiss_store")

print("Precomputation complete. The FAISS vector store has been saved to 'faiss_store'.")
