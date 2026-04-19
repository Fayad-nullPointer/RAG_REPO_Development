# 🧠 Text Embeddings & Chunking Strategies

Welcome to the **Embeddings Experiments** repository! This project serves as a sandbox for testing, evaluating, and optimizing text embeddings and chunking strategies. 

In Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) pipelines, the quality of your vector embeddings dictates the accuracy of your search results. This repo explores how different models and text-splitting techniques affect semantic retrieval.

## 📖 What are Embeddings?
Embeddings are mathematical representations of text (words, sentences, or entire documents) converted into high-dimensional arrays of numbers (vectors). 
Models are trained so that text with similar **semantic meaning** will have vectors that are closer together in the vector space. This enables systems to "understand" context rather than just keyword matching.

## 🎯 Core Concepts Explored

### 1. Embedding Models
Different use cases require different models. We experiment with:
* **Monolingual Models:** Fast and efficient (e.g., `all-MiniLM-L6-v2`). Best for English-only tasks.
* **Multilingual Models:** Optimized for cross-language alignment and non-English text (e.g., `intfloat/multilingual-e5-large`, `BAAI/bge-m3`).
* **Dimensionality:** Trading off between speed (lower dimensions like 384) and accuracy (higher dimensions like 1024+).

### 2. Text Chunking Strategies
Large documents must be split into "chunks" before they can be embedded. We analyze the impact of:
* **Chunk Size:** How many characters or tokens are included in a single vector. 
  * *Small Chunks (e.g., 100-300 characters):* Highly precise but risk losing surrounding context (fragmentation).
  * *Large Chunks (e.g., 700-1000 characters):* Great for context preservation but may suffer from "dilution," lowering similarity scores.
* **Chunk Overlap:** Including a sliding window of text to ensure context isn't lost at the boundaries of a split.

### 3. Vector Similarity Search
Evaluating how quickly and accurately we can retrieve the right vectors using:
* **FAISS** (Facebook AI Similarity Search)
* Distance metrics: **Inner Product** (Dot Product) and **Cosine Similarity**.

## 🛠️ Typical Tech Stack
To run embedding experiments, you generally need the following tools:
* **`sentence-transformers`**: Hugging Face's library for state-of-the-art sentence, text, and image embeddings.
* **`faiss-cpu` / `faiss-gpu`**: For lightning-fast similarity search and clustering of dense vectors.
* **`numpy` & `pandas`**: Data manipulation and array operations.

## 🚀 Quick Start Example

Here is a basic snippet demonstrating how to generate embeddings and run a semantic search:

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load your chosen embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 2. Your chunks of text
documents = [
    "How to reset my wireless router.",
    "Billing and subscription cancellation process.",
    "Slow internet speeds troubleshooting guide."
]

# 3. Create Embeddings (Normalize for Cosine Similarity)
embeddings = model.encode(documents, normalize_embeddings=True).astype(np.float32)

# 4. Build FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension) # Inner Product
index.add(embeddings)

# 5. Query and Retrieve
query = "Why is my Wi-Fi so slow?"
query_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
distances, indices = index.search(query_emb, k=1)

print(f"Best match: {documents[indices[0][0]]} (Score: {distances[0][0]:.4f})")
