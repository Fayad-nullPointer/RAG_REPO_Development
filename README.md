
```markdown
# Embedding Models & Chunk Size Optimization 🧠

This branch focuses on experimenting with and optimizing the retrieval phase of our RAG pipeline. The main goal is to evaluate different embedding models and chunk sizes to maximize retrieval accuracy for Arabic telecommunication queries (e.g., `"حل مشكلة بطء شبكة اﻻنترنت ؟؟"`).

## 🧪 Experiments Conducted

### 1. Model Evaluation
We tested three different embedding models to determine the best semantic understanding for non-English queries:
* `intfloat/multilingual-e5-large`
* `BAAI/bge-m3`
* `sentence-transformers/all-MiniLM-L6-v2`

### 2. Chunk Size Tuning
We evaluated character-based chunk sizes of **500, 600, and 700** to find the sweet spot between keyword density and context preservation.

## 📊 Findings & Report

* **BGE-M3 (`BAAI/bge-m3`)** emerged as the absolute best model. It demonstrated the deepest semantic understanding, moving past simple keyword matching to genuinely interpret the user's intent.
* **Multilingual E5 Large** performed decently but showed a bias towards specific keywords rather than the overall context.
* **All-MiniLM-L6-v2** struggled with Arabic word comprehension and context for our specific use case.

### The Effect of Chunk Size:
* **500 characters:** High precision for dense queries, but risks fragmenting the context (e.g., separating a symptom from its fix).
* **700 characters:** Excellent continuity and context mapping, though it slightly dilutes retrieval scores.
* **600 characters:** The optimal balance, maintaining adequate actionable steps without losing similarity accuracy.

**🏆 Conclusion:** The optimal configuration for this RAG pipeline is `BAAI/bge-m3` grouped with a chunk size of **600-700 characters**.

## 🚀 How to Run the Experiments
The tests and evaluations are fully documented in `rag-workshop-model-expriment.ipynb`.

1. Ensure dependencies are installed:
   ```bash
   pip install transformers sentence-transformers faiss-gpu-cu12 numpy
