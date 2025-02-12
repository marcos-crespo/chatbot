import streamlit as st
import os
import getpass

# -------------------------------
# Step 0: Environment Setup
# -------------------------------
# Ensure the API key is set (this prompt will only appear when running locally)
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# -------------------------------
# Step 1: Document Ingestion & Chunking
# -------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path to your PDF document (the Spanish Constitution)
pdf_path = "documents/constitution.pdf"

# Load the PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()  # Returns a list of Document objects

# Use a text splitter that splits on "\nArticle"
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Adjust based on your LLM's token limit
    chunk_overlap=100,    # Overlap to maintain context
    separators=["\nArticle"]
)

# Split the documents into chunks
chunks = text_splitter.split_documents(documents)
st.write(f"Total chunks created: {len(chunks)}")

# -------------------------------
# Step 2: Embedding and Vector Store Setup
# -------------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize the embedding model (using all-mpnet-base-v2)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create a FAISS vector store from the chunks
vector_store = FAISS.from_documents(chunks, embedding_model)

# -------------------------------
# Step 3: Build the RetrievalQA Chain
# -------------------------------
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA

# Initialize the LLM (using gpt-4o-mini via OpenAI in this example)
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Create a retriever from the vector store (fetch top 3 most similar chunks)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Build the RetrievalQA chain that stitches together retrieval and LLM generation.
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# -------------------------------
# Step 4: Streamlit Frontend
# -------------------------------
st.title("Spanish Constitution Q&A Chatbot")
st.write("Ask any question about the Spanish Constitution.")

# Input field for user query
user_query = st.text_input("Your question:")

if st.button("Get Answer") and user_query:
    with st.spinner("Generating answer..."):
        answer = qa_chain.run(user_query)
    st.subheader("Answer:")
    st.write(answer)
