# NLP-Final-Project

This is a final project for the Natural Language Processing class at Northeastern University, Fall 2025

This project implements a fact verification system for the HoVer dataset using two different retrieval methods:
1.  **BM25 Retrieval**: A traditional sparse retrieval method using Elasticsearch.
2.  **Dense Retrieval**: An experimental dense retrieval method using Sentence-BERT and FAISS.

## Overview

The system performs multi-hop fact verification in three stages:
1.  **Document Retrieval**: Find relevant Wikipedia articles using either BM25 or Dense Retrieval.
2.  **Sentence Selection**: Extract specific sentences from retrieved documents.
3.  **Claim Verification**: Classify claims as SUPPORTED or NOT_SUPPORTED.

## Project Structure
```bash
Project/
├── src/
│   ├── bm25_retriever/
│   │   └── hover_project.py      # Main implementation (WikipediaIndexer, BM25Retriever)
│   └── dense_retriever/
│       └── dense_retrieval_faiss.py # Dense retrieval implementation
├── scripts/
│   ├── run_bm25_indexing.py      # Script to index Wikipedia into Elasticsearch
│   └── run_bm25_retrieval.py     # Script to retrieve documents for HoVer claims with BM25
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── dense_retrieval_faiss.ipynb
│   └── DenseRetr.ipynb
├── data/
│   ├── hover_train_release_v1.1.json
│   ├── hover_dev_release_v1.1.json
│   ├── hover_test_release_v1.1.json
│   └── enwiki-20171001-pages-meta-current-withlinks-processed/
│       └── ...
├── reports/
│   ├── CS6120_Report.pdf
│   ├── DenseRetr.ipynb - Colab.pdf
│   └── Presentation.pdf
├── output/
│   ├── hover_train_bm25_top100.json
│   ├── hover_dev_bm25_top100.json
│   └── hover_test_bm25_top100.json
├── requirements.txt
└── README.md
```

## Prerequisites

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Docker Desktop

- **Download**: https://www.docker.com/products/docker-desktop/
- Install and launch Docker Desktop
- Wait for Docker to start (green indicator in menu bar/system tray)

### 3. Download HoVer Dataset

Download the HoVer dataset files and place them in the `data/` folder:
- `hover_train_release_v1.1.json`
- `hover_dev_release_v1.1.json`
- `hover_test_release_v1.1.json`

**Source**: [HoVer Dataset](https://github.com/hover-nlp/hover)

### 4. Download HotpotQA Wikipedia Dump

**Important**: This is a large download (~13 GB compressed, ~30 GB extracted)

1. Download from: https://hotpotqa.github.io/wiki-readme.html
   - File: `enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2`

2. Extract the archive:
```bash
   tar -xjf enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
```

3. Place the extracted folder in `data/`:
```
   data/enwiki-20171001-pages-meta-current-withlinks-processed/
```

The folder should contain subdirectories `AA/`, `AB/`, ..., `FZ/`, each with `.bz2` files.

## Setup and Execution: BM25 Retrieval

### Step 1: Start Elasticsearch
```bash
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  elasticsearch:7.17.0
```

Wait ~30 seconds for Elasticsearch to start, then verify:
```bash
curl http://localhost:9200
```

You should see JSON output with Elasticsearch version info.

### Step 2: Index Wikipedia Documents

**One-time operation** (~30-60 minutes depending on hardware)
```bash
python3 scripts/run_bm25_indexing.py
```

This will:
- Create an Elasticsearch index named `hotpot_wiki`
- Index ~5.4 million Wikipedia documents
- Build BM25 inverted index for fast retrieval

**Note**: The index persists in Elasticsearch, so you only need to do this once. If you restart Docker, use `docker start elasticsearch` to resume.

### Step 3: Run BM25 Retrieval

**Retrieves top-100 documents for each claim** (~20-30 minutes total)
```bash
python3 scripts/run_bm25_retrieval.py
```

This will:
- Process train, dev, and test sets
- Retrieve top-100 Wikipedia documents per claim using BM25
- Save results to `output/hover_{split}_bm25_top100.json`
- Evaluate retrieval quality (coverage and recall)

## Setup and Execution: Dense Retrieval (Experimental)

This project also includes an experimental dense retrieval system using Sentence-BERT and FAISS. The code for this is in `src/dense_retriever/` and `notebooks/`.

### Step 1: Generate FAISS Index

You can generate a FAISS index of the Wikipedia dump using the `dense_retrieval_faiss.ipynb` notebook. This will process the Wikipedia articles, encode them into vectors using a Sentence-BERT model, and save a FAISS index to disk.

### Step 2: Run Retrieval

The `DenseRetr.ipynb` notebook contains code for running retrieval using the generated FAISS index.

## Results

The following tables summarize the performance of the different retrieval and verification models.

### BM25 Retrieval

| Metric | Value |
| :--- | :--- |
| Recall@10 | 33.76% |
| Recall@100 | 51.7% |

### BM25 + Verification

| Metric | Value |
| :--- | :--- |
| Best Accuracy | ~51-52% |
| Best F1-Score | ~55% |
| Optimal Threshold | 0.05 |

### BM25 + Dense Retrieval (Re-ranking) + Verification

| Metric | Value | Change from BM25 |
| :--- | :--- | :--- |
| **Retrieval** | | |
| Recall@10 | 43.24% | +9.48pp |
| Coverage@10 | 83.97% | |
| **Verification** | | |
| Best Accuracy | ~52.5% | +0.5pp |
| Best F1-Score | ~57% | +2pp |
| Optimal Threshold | 0.05 | |

## Docker Commands Reference
```bash
# Check if Elasticsearch is running
docker ps

# View Elasticsearch logs
docker logs elasticsearch

# Stop Elasticsearch
docker stop elasticsearch

# Start Elasticsearch again (after stopping)
docker start elasticsearch

# Remove container (will delete index!)
docker stop elasticsearch
docker rm elasticsearch
```