import math
import pickle
import re
from collections import defaultdict
import nltk
import numpy as np
from flask import Flask, request, jsonify
from google.cloud import storage
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.porter import *
from sentence_transformers import SentenceTransformer, util
from typing import List
import pandas as pd
import os
import heapq

nltk.download('stopwords')
client = storage.Client()

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# consts
project_id = 'ir-final-project-12016'
data_bucket_name = 'ir-final-project-bucket'
page_view_map_filename = "pageviews-202108-user.pkl"
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
stemmer = PorterStemmer()
storage_output_folder = "prod"
MIN_TOKEN_LEN = 3


def download_df(prefix):
    blobs = storage.Client().bucket(data_bucket_name).list_blobs(prefix=prefix)
    for blob in blobs:
        if blob.name.endswith("/"): 
            continue
        rel_path = os.path.relpath(blob.name, start=os.path.dirname(prefix))
        local_file_path = os.path.join(".", rel_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        print(f"Downloading {rel_path}...")
        blob.download_to_filename(local_file_path)

def load_resources():
    """
    Downloads and loads all necessary metadata, statistics, and mappings 
    required for the search engine's ranking and retrieval logic.
    """

    # download and load statistical data for different fields
    files = ["anchor_stats.pkl", "title_stats.pkl", "body_stats.pkl"]
    for file_name in files:
        client.bucket(data_bucket_name).blob(f"{storage_output_folder}/stats/{file_name}.pkl").download_to_filename(file_name)
    with open('body_stats.pkl', 'rb') as f:
        body_data = pickle.load(f)
    with open('title_stats.pkl', 'rb') as f:
        title_data = pickle.load(f)
    with open('anchor_stats.pkl', 'rb') as f:
        anchor_data = pickle.load(f)

    # download and load PageRank scores
    client.bucket(data_bucket_name).blob(f"{storage_output_folder}/mappings/pagerank_dict.pkl").download_to_filename("pagerank_dict.pkl")
    with open("pagerank_dict.pkl", "rb") as f:
        pagerank_dict = pickle.load(f)

    # download and load PageView statistics
    client.bucket(data_bucket_name).blob(f"{storage_output_folder}/mappings/pageview_dict.pkl").download_to_filename("pageview_dict.pkl")
    with open("pageview_dict.pkl", "rb") as f:
        wid2pv = pickle.load(f)

    # downloads a Parquet file mapping Document IDs to their actual string titles
    prefix = f"{storage_output_folder}/titles_df"
    download_df(prefix)
    df = pd.read_parquet("titles_df", engine='pyarrow')
    title_df = df.set_index("id")["title"]

    # document length is essential for normalizing scores
    dl_by_field = dict()
    for field_name in ["anchor", "body", "title"]:
        prefix = f"{storage_output_folder}/stats/dl_{field_name}.parquet"
        download_df(prefix)
        curr = pd.read_parquet(f"dl_{field_name}.parquet", engine='pyarrow')
        curr.set_index("id")["len"]
        dl_by_field[field_name] = curr

    return body_data, title_data, anchor_data, title_df, pagerank_dict, wid2pv, dl_by_field


# read the inverted index from storage
def read_inverted_index(index_name: str):
    # read the inverted index from storage
    index_dst = f"{index_name}.pkl"
    client.bucket(data_bucket_name).blob(f"{storage_output_folder}/inverted_indecies/{index_name}.pkl").download_to_filename(
        index_dst
    )
    with open(index_dst, "rb") as f:
        return pickle.load(f)


#load the inverted indicies
body_inverted_index = read_inverted_index("body_inverted_index")
title_inverted_index = read_inverted_index("title_inverted_index")
inverted_anchor = read_inverted_index("ancher_inverted_index")

# smart indecies
title_inverted_index_smart = read_inverted_index("title_inverted_index_smart")
anchor_inverted_index_smart = read_inverted_index("ancher_inverted_index_smart")

BODY_STATS, TITLE_STATS, ANCHOR_STATS, TITLES_DF, PAGERANK_DICT, WID2PV, DL_BY_FIELD = load_resources()

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def tokenize_smart(text):
    """
    Advanced tokenization: Downcasing, Stopword removal, Stemming, and Bigrams.
    Used for 'Title' field to capture high-intent phrases.
    """
    tokens = [tok.group().lower() for tok in RE_WORD.finditer(text)]
    tokens = [tok for tok in tokens if tok not in all_stopwords]
    tokens = [stemmer.stem(tok) for tok in tokens]
    bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def search_helper(query: str, inverted_index):
    """
    Ranks documents by the count of distinct query terms present.
    Often used as a fallback or for anchor-text-only searches.
    """
    query_tokens = {
        tok.group().lower()
        for tok in RE_WORD.finditer(query)
        if tok.group().lower() not in all_stopwords
    }
    if not query_tokens:
        return []

    # Map doc_id -> set of query tokens that appear in its anchor text
    doc_seen_per_token = defaultdict(set)

    # Only process posting lists for terms in the query
    for term in query_tokens:
        posting_list = inverted_index.read_a_posting_list('.', term, data_bucket_name)
        for doc_id, tf in posting_list:
            doc_seen_per_token[doc_id].add(term)
    
    # Count distinct query words per document
    doc2count = [(doc_id, len(token_set)) for doc_id, token_set in doc_seen_per_token.items()]

    # Sort by number of distinct query words descending
    doc2count.sort(key=lambda x: -x[1])
    if len(doc2count) == 0:
        return []
    res = []
    for doc_id, count in doc2count:
        title = TITLES_DF.get(int(doc_id), "Unknown")
        res.append((str(doc_id), title))
    return res

def tokenize_text(text_element, tokenize_func):
    if isinstance(text_element, list):
        if not text_element:
            return []
        texts = []
        for r in text_element:
            if hasattr(r, "text") and isinstance(r.text, str):
                texts.append(r.text)
        if not texts:
            return []
        full_text = " ".join(texts)
    else:
        full_text = text_element

    return tokenize_func(full_text)

def simple_tokenize(full_text: str) -> List[str]:
    """Standard tokenization: Downcasing and stopword removal only."""
    return  [
        tok.group().lower()
        for tok in RE_WORD.finditer(full_text)
        if tok.group().lower() not in all_stopwords
    ]

def tokenize_anchor(text):
    stemmer = PorterStemmer()
    return [
        stemmer.stem(tok.group().lower())
        for tok in RE_WORD.finditer(text)
        if tok.group().lower() not in all_stopwords
        and len(tok.group()) >= MIN_TOKEN_LEN
    ]

def get_bm25_scores(query: str):
    """
    Calculates the BM25 score across multiple inverted indices.
    Weights are tuned to favor Title and Anchor matches over Body matches.
    """
    k1, b = 1.5, 0.75 # BM25 Hyperparameters
    candidate_scores = defaultdict(float)

    # Configuration for multi-field search
    fields = [
        (body_inverted_index, BODY_STATS, 0.1, simple_tokenize, DL_BY_FIELD["body"]),  # Low noise
        (title_inverted_index_smart, TITLE_STATS, 0.6, tokenize_smart, DL_BY_FIELD["title"]), # Max Signal
        (anchor_inverted_index_smart, ANCHOR_STATS, 0.3, tokenize_anchor, DL_BY_FIELD["anchor"]) # High Signal
    ]

    for index, stats, weight, tokenize_func, dl_df in fields:
        # Preprocess query specifically for the current field's requirements
        query_tokens = tokenize_text(query, tokenize_func)
        unique_tokens = np.unique(query_tokens)

        avgdl = stats['avgdl']

        for term in unique_tokens:
            n_ti = index.df.get(term, 0) # Number of docs containing the term
            if n_ti == 0:
                continue

            # Inverse Document Frequency: Rare words get higher scores
            idf = math.log(1 + (len(TITLES_DF) - n_ti + 0.5) / (n_ti + 0.5))

            try:
                # Retrieve list of [doc_id, frequency] from the inverted index
                pls = index.read_a_posting_list('.', term, data_bucket_name)
                for doc_id, freq in pls:
                    doc_len = dl_df.get(doc_id, avgdl)

                    # BM25 core formula
                    denominator = freq + k1 * (1 - b + b * (doc_len / avgdl))
                    score = idf * (freq * (k1 + 1)) / denominator

                    # Accumulate weighted score across all fields
                    candidate_scores[doc_id] += (score * weight)
            except Exception:
                continue
    return candidate_scores


def get_semantic_similarity(query, titles):
    """
    Computes the semantic overlap between a search query and a list of titles
    using Vector Embeddings and Cosine Similarity.
    """
    if not titles:
        return []
    
    # The 'semantic_model' captures the context and meaning of the words
    query_emb = semantic_model.encode(query, convert_to_tensor=True)
    title_embs = semantic_model.encode(titles, convert_to_tensor=True)

    # score closer to 1.0 means the query and title are semantically very similar
    semantic_sims = util.cos_sim(query_emb, title_embs)[0].tolist()
    return semantic_sims


def min_max_normalize(values):
    """
    Scales a list of scores into a fixed range of [0, 1].
    This is essential before 'Ensemble Ranking' (combining different scoring signals).
    """
    min_val = min(values)
    max_val = max(values)
    # Handle the edge case where all scores are the same
    if max_val == min_val:
        return [0.5] * len(values)
    
    # Standard Min-Max Scaling formula: (value - min) / (max - min)
    return [(x - min_val) / (max_val - min_val) for x in values]


def rank_candidates(query, top_candidates_scores):
    # extract raw data
    candidate_ids = [int(cid) for cid, _ in top_candidates_scores]
    titles = [TITLES_DF.get(cid, "Unknown") for cid in candidate_ids]
    
    # compute Raw Feature Vectors
    bm25_scores = [score for _, score in top_candidates_scores]
    
    # Semantic Vector (Title Similarity)
    semantic_scores = get_semantic_similarity(query, titles)
    
    # PageRank Vector (Log-transformed to handle power-law distribution)
    pr_scores = [math.log(PAGERANK_DICT.get(cid, 1e-6) + 1e-10) for cid in candidate_ids]
    
    # PageViews Vector (Log-transformed)
    pv_scores = [math.log(WID2PV.get(cid, 0) + 1) for cid in candidate_ids]

    # normalize Features to [0, 1]
    bm25_norm = min_max_normalize(bm25_scores)
    sem_norm = min_max_normalize(semantic_scores)
    pr_norm = min_max_normalize(pr_scores)
    pv_norm = min_max_normalize(pv_scores)

    # weighted Combination
    final_ranked_list = []
    for i, cid in enumerate(candidate_ids):
        final_score = (
            (bm25_norm[i] * 0.7) +  (sem_norm[i]  * 0.2) +   (pv_norm[i]   * 0.05) +   (pr_norm[i]   * 0.05)    
        )
        final_ranked_list.append((str(cid), titles[i], final_score))

    return sorted(final_ranked_list, key=lambda x: x[2], reverse=True)

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_tokens = tokenize_smart(query)
    if query_tokens:
        # get initial scores for all documents containing query terms
        all_bm25_scores = get_bm25_scores(query)
        if all_bm25_scores:
            # use a heap to efficiently find top 200 scores (O(N log 200))
            top_candidates = heapq.nlargest(200, all_bm25_scores.items(), key=lambda x: x[1])
            # pass candidates to the ranking function
            final_results = rank_candidates(query, top_candidates)
            # Return the top 100 as (wiki_id, title) tuples
            res = [(res[0], res[1]) for res in final_results[:100]]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = search_helper(query, body_inverted_index)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = search_helper(query, title_inverted_index)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = search_helper(query, inverted_anchor)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # defaults to 0.0 if the ID is not found
    res = [PAGERANK_DICT.get(int(wid), 0.0) for wid in wiki_ids]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # defaults to 0 if the ID is not found
    res = [WID2PV.get(int(wid), 0) for wid in wiki_ids]
    # END SOLUTION
    return jsonify(res)


def run(**options):
    app.run(**options)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
