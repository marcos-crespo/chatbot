import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
import openai

@st.cache_data
def load_vector_store():
    # This assumes you have run your precomputation script and saved the store locally.
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.load_local("faiss_store", embedding_model, allow_dangerous_deserialization=True)
    return vector_store

vector_store = load_vector_store()
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# -------------------------------
# Sidebar: API Key Input
# -------------------------------
with st.sidebar:
    st.header("API Settings")
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"

# -------------------------------
# Main Page Header
# -------------------------------
# Title with two sun emojis (you can adjust emoji style if needed)
st.title("☀️☀️**Spanish Constitution Q&A**")
# Subtitle with a clickable link to the official document
st.markdown("**Ask questions about the [Spanish Constitution official document](https://www.tribunalconstitucional.es/es/tribunal/normativa/normativa/constitucioningles.pdf)**")

# -------------------------------
# Question Input Section
# -------------------------------
user_query = st.text_input("Your question:")

# Initialize session state for history if it doesn't exist
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Get Answer") and user_query:
    if not api_key:
        st.markdown("**Missing API key**")
    else:
        try:
            llm = init_chat_model("gpt-4o-mini", model_provider="openai")
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            # Attempt to get the answer from the RetrievalQA chain
            answer = qa_chain.invoke(user_query)
            # Also retrieve the context chunks if needed
            retrieved_chunks = retriever.get_relevant_documents(user_query)
        except openai.AuthenticationError:
            # Catch the invalid API key error and show an error message
            st.error("Invalid API key")
        except Exception as e:
            # Optionally catch other exceptions and display a generic error
            st.error(f"An error occurred: {str(e)}")
        else:
            # If no error, append the result to history and display the answer
            st.session_state.history.append({
                "query": user_query,
                "answer": answer,
                "retrieved": retrieved_chunks
            })
            st.markdown("### Answer:")
            st.write(answer['result'])

# -------------------------------
# Side-by-Side: Example Questions and History
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Example Questions")
    sample_questions = [
        "What does Article 10 say about freedom?",
        "How is power distributed according to Article 5?",
        "What rights are mentioned in Article 1?",
    ]
    for question in sample_questions:
        st.write(f"• {question}")

with col2:
    st.subheader("History of Questions")
    if st.session_state.history:
        # Display history in chronological order (1 = oldest)
        for idx, entry in enumerate(st.session_state.history, start=1):
            st.markdown(f"**{idx}. Query:** {entry['query']}")
            st.markdown(f"**Answer:** {entry['answer']}")
            st.markdown("---")
    else:
        st.write("No history yet.")
