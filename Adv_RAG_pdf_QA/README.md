Advanced RAG Pipeline with Hybrid Retrieval & Automated Evaluation
A production-style Retrieval-Augmented Generation (RAG) system built over PDF documents, featuring hybrid retrieval, reranking, query transformation, and automated RAGAS evaluation.

📊 Evaluation Results (RAGAS)
MetricScoreFaithfulness0.80Answer Relevancy0.69Context Precision0.83Context Recall1.00

# RAG‑Based Document Q&A System

Hybrid retrieval (dense + sparse) + reranking + HyDE over PDF documents. Built with LlamaIndex, ChromaDB, quantized Llama‑2‑7B, and evaluated with RAGAS.

## Features
- PDF parsing via LlamaParse → Markdown chunks
- Hybrid retrieval: Vector (BGE‑small‑EN) + BM25 + Reciprocal Rank Fusion
- Cross‑encoder reranking (FlagEmbedding) and HyDE query transformation
- 4‑bit quantized Llama‑2‑7B (BitsAndBytes NF4)
- Persistent ChromaDB vector store
- Automated evaluation with RAGAS (faithfulness, context recall, precision)

## Results on Financial PDFs
| Metric             | Score|
|--------------------|------|
| Context Recall     | 1.00 |
| Context Precision  | 0.83 |
| Faithfulness       | 0.80 |
| Answer Relevancy   | 0.69 |

## Setup
```bash
pip install -r requirements.txt

# Pipeline Architecture
PDF Documents
     ↓
LlamaParse (structured markdown extraction)
     ↓
MarkdownNodeParser (semantic chunking)
     ↓
BGE-small Embeddings (BAAI/bge-small-en-v1.5)
     ↓
ChromaDB (persistent vector store)
     ↓
Hybrid Retrieval
├── Dense: Vector search (semantic)
└── Sparse: BM25 (keyword)
     ↓
Reciprocal Rank Fusion (RRF)
     ↓
FlagEmbedding Reranker (BAAI/bge-reranker-base)
     ↓
HyDE Query Transformation
     ↓
Llama-2-7B (4-bit NF4 quantized)
     ↓
RAGAS Automated Evaluation
⚙️ Key Components

LlamaParse — Structured PDF ingestion with markdown output for clean chunking
Hybrid Retrieval — Combines dense vector search with BM25 sparse retrieval for better coverage
Reciprocal Rank Fusion — Merges and reranks results from both retrievers
FlagEmbedding Reranker — Cross-encoder reranking to surface the most relevant chunks
HyDE — Hypothetical Document Embeddings to improve query-document alignment
4-bit Quantization — NF4 quantized Llama-2-7B via BitsAndBytes for memory-efficient inference
RAGAS Evaluation — Automated testset generation and evaluation using Groq (llama-3.1-8b) + HuggingFace embeddings

# How to Run

Open the notebook in Google Colab (recommended — requires GPU)
Add the following to Colab Secrets:

HF_TOKEN — from huggingface.co
LLAMA_CLOUD_API_KEY — from cloud.llamaindex.ai
GROQ_API_KEY — from console.groq.com


Upload your PDF files when prompted
Run all cells top to bottom

📋 Requirements
See requirements.txt. All installs are handled inside the notebook for Colab compatibility.
⚠️ Notes

Designed to run on Google Colab with T4 GPU
Free Colab handles ~10–30 medium PDFs comfortably before RAM limits apply
RAGAS testset is limited to 1 sample by default to stay within Groq free tier TPM limits — increase testset_size for larger evaluations
Never commit your data/ folder or chroma_db/ to GitHub — add them to .gitignore

## 📸 Output Screenshots

### 🔹 Example 1
![Output 1](124153.png)

### 🔹 Example 2
![Output 2](130642.png)

🛠️ Tech Stack
Python · LlamaIndex · ChromaDB · LlamaParse · HuggingFace Transformers · BitsAndBytes · RAGAS · LangChain · Groq · sentence-transformers · FlagEmbedding
