# Wikipedia Search Engine - Information Retrieval Final Project

This project implements a scalable search engine for English Wikipedia using PySpark on Google Cloud Platform (GCP). It features a multi-stage indexing pipeline, a custom inverted index implementation, and a Flask-based frontend that performs hybrid ranking using BM25, Semantic Search, PageRank, and PageViews.

## 📂 Code Structure & Organization

The codebase is organized into three main components:
1. **`inverted_index_gcp.py`**: The core data structure logic.
2. **`final_project_create_all_data.ipynb`**: The ETL (Extract, Transform, Load) and Indexing pipeline.
3. **`search_frontend.py`**: The runtime search API and ranking engine.

---
### 1. `inverted_index_gcp.py`
**Role:** Core Library

This file defines the `InvertedIndex` class, which handles the writing, reading, and storage of posting lists on Google Cloud Storage (GCS).

* **InvertedIndex Class**:
    * Manages Document Frequency (`df`) and Term Total counts.
    * **`write_index`**: Serializes the global statistics (like `df`) to disk.
    * **`read_a_posting_list`**: Efficiently fetches posting lists for specific query terms without loading the entire index into memory.

* **Binary Storage (MultiFileWriter / MultiFileReader)**:
    * Implements custom binary I/O to store posting lists in sharded bin files (block size ~2MB).
    * Compresses DocID and TF (Term Frequency) into 6-byte tuples to optimize storage space and I/O throughput.

### 2. `final_project_create_all_data.ipynb`
**Role:** Data Pipeline (ETL)

This Jupyter Notebook runs on a GCP Dataproc cluster. It processes raw Wikipedia dumps and generates the artifacts required by the search engine.

**Key Operations:**

* **PageViews Processing**:
    * Parses the `pageviews-202108-user` dump.
    * Creates a dictionary mapping `wiki_id` to monthly view counts.

* **PageRank Calculation**:
    * Uses GraphFrames to construct a link graph from Wikipedia anchor text data.
    * Computes PageRank to measure article authority.

* **Inverted Index Construction**:
    * Builds three separate indices using PySpark RDDs:
        1. **Body Index**: Standard tokenization (Stopword removal, lowercasing).
        2. **Title Index**: "Smart" tokenization (Stemming + Bigrams + Filter by word length) to capture high-intent queries.
        3. **Anchor Index**: Stemmed tokenization + Filter by word length of text from incoming links.
    * Calculates Document Lengths (`dl`) for every document in all three fields (crucial for BM25 normalization) and saves them as Parquet files.

* **Title Mapping**:
    * Extracts `(id, title)` pairs to allow the frontend to return human-readable titles.

### 3. `search_frontend.py`
**Role:** Application Server (Flask)

This script runs the search engine. Upon startup, it downloads all indices, stats, and models from GCS into memory for fast retrieval.

**Key Functionalities:**

* **Resource Loading**:
    * Downloads pickles (`.pkl`) for PageRank, PageViews, and field statistics.
    * Downloads the inverted indices and connects to the binary posting files.
    * Loads the SentenceTransformer model (`all-MiniLM-L6-v2`) for semantic analysis.

* **Query Processing**:
    * Replicates the tokenization logic used in the notebook (Simple vs. Smart/Stemming) to ensure query terms match index terms.

* **Ranking Algorithm (`/search` endpoint)**:
    1. **BM25 (Textual Match)**: Calculates a weighted BM25 score across three fields:
        * Title (Weight: 0.6)
        * Anchor (Weight: 0.3)
        * Body (Weight: 0.1)
    2. **Heap Selection**: Efficiently selects the top 200 candidates based on BM25.
    3. **Semantic Similarity**: Computes Cosine Similarity between the query embedding and the candidate titles.
    4. **Hybrid Re-Ranking**: Computes a final score using a weighted formula:

    ```python
    Final_Score = (BM25 * 0.7) + (Semantic * 0.2) + (PageViews * 0.05) + (PageRank * 0.05)
    ```

* **API Endpoints**:
    * `/search`: Main hybrid search.
    * `/search_body`, `/search_title`, `/search_anchor`: Targeted field searches using simple counts.
    * `/get_pagerank`, `/get_pageview`: Metadata retrieval.

---

## 📊 Ranking Logic Summary

The engine prioritizes relevance using a "Cascading" approach:

1. **Recall**: Uses Inverted Indices to find all documents containing query terms.
2. **Initial Ranking**: Uses BM25F (Multifield BM25) to score documents based on term frequency and document length, heavily favoring matches in Titles and Anchors.
3. **Refinement**: The top 200 results are re-ranked by combining:
    * Textual Relevance: (BM25)
    * Semantic Relevance: (BERT embeddings of Query vs Title)
    * Popularity: (PageViews)
    * Authority: (PageRank)

This ensures that results are not only textually accurate but also authoritative and semantically meaningful.