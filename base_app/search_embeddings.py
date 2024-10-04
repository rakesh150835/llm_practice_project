import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')



def load_faiss_index():
    return faiss.read_index('vectorstore/faiss_index.bin')


def load_embeddings(embeddings_folder='vectorstore'):
    # Load filenames from the saved .npy file
    filenames = np.load(os.path.join(embeddings_folder, 'filenames.npy'))

    embeddings = []
    # Iterate over the range of the number of embeddings based on the length of the filenames
    for i in range(len(filenames)):
        embedding_file = os.path.join(embeddings_folder, f'embeddings_{i}.npy')
        if os.path.exists(embedding_file):
            # Load the corresponding embedding
            embedding = np.load(embedding_file)
            embeddings.append(embedding)
    # Convert embeddings to a numpy array for FAISS
    embeddings = np.vstack(embeddings)  # Stack embeddings vertically

    return embeddings, filenames




def embed_question(question):
    return model.encode([question])


def search_similar(question_embedding, index, top_k=1):
    distances, indices = index.search(question_embedding, top_k)
    return distances, indices


def get_related_text(indices, filenames):
    related_files = [filenames[i] for i in indices[0] if i < len(filenames)]
    return related_files


def generate_concise_answer(question, context):
    generator = pipeline("text-generation", model="gpt2")
    prompt = f"Provide a clear, specific, and concise answer strictly based on the information given in the context and the question. Avoid introducing any details or assumptions beyond the context provided. If the context does not directly address the question, base your response only on what is available in the context. If context is not availabel just reply you don't have enough information to give proper response please provide more details.\n\nQuestion: {question}\nContext: {context}\nAnswer:"

    response = generator(prompt, max_length=100, num_return_sequences=1)
    generated_answer = response[0]['generated_text'].split("Answer:")[-1].strip()

    return generated_answer