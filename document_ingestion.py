from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Specify the path to your PDF document
pdf_path = "documents/constitution.pdf"

# Load the PDF using LangChain's PyPDFLoader
loader = PyPDFLoader(pdf_path)
documents = loader.load()  # This returns a list of Document objects
# Print the extracted text to look for common patterns (like paragraph breaks or article headers)
# for i, doc in enumerate(documents):
#     print(f"--- Page {i+1} ---")
#     print(doc.page_content)
#     print("\n")
# Clean and chunk the text using a text splitter.
# The RecursiveCharacterTextSplitter will split text into chunks with a specified size,
# with some overlap to maintain context.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Adjust based on your LLM's max token limit
    chunk_overlap=100,    # Overlap between chunks to preserve context
    separators=["\nArticle"]
)

# Split the loaded documents into smaller chunks
chunks = text_splitter.split_documents(documents)

print(len(chunks))
# Print out the first few chunks for inspection
for i, chunk in enumerate(chunks[:12]):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content)
    print("\n")