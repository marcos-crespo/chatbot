import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Define your corpus of curated music theory texts.
# In a real scenario, you’d load a more extensive, curated collection.
documents = [
    "Music theory is the study of how music works. It includes harmony, melody, rhythm, and structure.",
    "The circle of fifths is a visual representation of the relationships among the 12 tones of the chromatic scale.",
    "Chord progressions are a series of musical chords played in sequence, forming the harmonic backbone of a song.",
    "Scales form the basis of melody and harmony. Major and minor scales are the most common, but modes and exotic scales add color.",
]

# 2. Load an embedding model (using SentenceTransformer)
embedder = SentenceTransformer('all-mpnet-base-v2')

# Embed your documents
doc_embeddings = embedder.encode(documents, convert_to_tensor=False)
doc_embeddings = np.array(doc_embeddings).astype('float32')

# 3. Build a FAISS index using L2 (Euclidean) distance.
embedding_dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(doc_embeddings)

def retrieve_documents(query, top_k=3):
    """Retrieve the top_k documents most similar to the query."""
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

# 4. Set up the generative model.
# For demo purposes, we use a smaller GPT-Neo model; for production, consider fine-tuning a larger or domain-adapted model.
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
generator_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

def generate_response(query, retrieved_docs, max_new_tokens=150):
    """
    Combine the retrieved context with the user query to form a prompt,
    and generate an answer using the language model.
    """
    # Construct the prompt with context and the query.
    context = "\n".join(retrieved_docs)
    prompt = (
        "You are a music theory assistant. Answer the question based on the context provided.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = generator_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the answer part by removing the prompt from the generated text.
    answer = generated_text[len(prompt):].strip()
    return answer

def chat(query):
    retrieved_docs = retrieve_documents(query)
    answer = generate_response(query, retrieved_docs)
    return answer

if __name__ == "__main__":
    print("Music Theory Chatbot (type 'quit' to exit)")
    while True:
        user_input = input("\nEnter your music theory question: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        response = chat(user_input)
        print("\nResponse:", response)
