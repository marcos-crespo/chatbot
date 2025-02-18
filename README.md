# Spanish Constitution Q&A ChatBot

[Open in Streamlit](https://chatbot-4g5ea76rwyjj9e9biaa49y.streamlit.app/)

[Medium Article on the How To Build your First RAG](https://medium.com/@marcoscrespo-ai/building-your-first-rag-c0a83ae9a6db)

## Overview

The **Spanish Constitution Q&A ChatBot** is RAG weekend project that lets users ask questions about the official Spanish Constitution document. The app uses LangChain to ingest and chunk the document, creates embeddings with a HuggingFace model, and builds a FAISS vector store for fast retrieval. An LLM (e.g., GPT-4O-mini via OpenAI) then generates answers based on the retrieved context.
The main goal of this app was to learn how to build a simple RAG and to add those technologies to my portfolio.
## Features

- **Document Processing:** Ingests the official Spanish Constitution PDF and splits it into manageable text chunks.
- **Embeddings & Retrieval:** Uses `sentence-transformers/all-mpnet-base-v2` to compute embeddings and stores them in a FAISS index.
- **Q&A System:** Leverages LangChain's RetrievalQA chain to fetch relevant text and generate answers.
- **User-Friendly Interface and deployment** Features a clean UI built with Streamlit, including:
  - A sidebar for entering your OpenAI API key. **MANDATORY**
  - A question box and button to submit queries.
  - A two-column layout displaying example questions and the history of queries.
  - Easily deployable on Streamlit Cloud with continuous integration from GitHub.

## Demo

Try the live demo on [Streamlit Cloud](https://chatbot-4g5ea76rwyjj9e9biaa49y.streamlit.app/).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Precompute the Vector Store:**

   Run the precomputation script to ingest the PDF, chunk the text, and save the FAISS index. 

   ```bash
   python build_index.py
   ```

## Configuration

- **OpenAI API Key:**

  The app requires an OpenAI API key. You can:
  
  - Enter your API key in the sidebar when running the app locally. **Recommended**. You can create an OpenAI API key [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key).
  - **Or**, create a `.streamlit/secrets.toml` file with the following content for Streamlit Cloud deployment:

    ```toml
    [general]
    OPENAI_API_KEY = "your-api-key-here"
    ```

## Running the App

To run the app locally, use:

```bash
streamlit run app.py
```

## Usage

1. **Enter Your API Key:**  
   Provide your OpenAI API key in the sidebar.

2. **Ask a Question:**  
   Type a question in the text box (e.g., "What does Article 10 say about freedom?") and click **Get Answer**.

3. **Review the Response:**  
   The app displays the generated answer below the question box, and your query along with its answer is added to the query history.

## Project Structure

- `app.py`: Main Streamlit application.
- `build_index.py`: Script to ingest, chunk, and precompute the FAISS vector store.
- `requirements.txt`: Lists all dependencies.
- `documents`: folder with the 'constituion.pdf'.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any doubts please contact me at: marcoscrespo.diaz@gmail.com
