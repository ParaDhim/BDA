import re
import binascii
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Tuple, Set, Dict
import os
from six.moves import urllib

# ------------------- Data Fetching -------------------
def fetch_data(download_root="https://raw.githubusercontent.com/chrisjmccormick/MinHash/master/data",
               plagiarism_path="datasets/plagiarism",
               data_sizes=[100, 1000, 2500, 10000],
               maxsize=1000):
    if not os.path.isdir(plagiarism_path):
        os.makedirs(plagiarism_path)
    for size in data_sizes:
        if size <= maxsize:
            train_file = f"articles_{size}.train"
            train_path = os.path.join(plagiarism_path, train_file)
            if not os.path.exists(train_path):
                train_url = f"{download_root}/{train_file}"
                urllib.request.urlretrieve(train_url, train_path)
            
            truth_file = f"articles_{size}.truth"
            truth_path = os.path.join(plagiarism_path, truth_file)
            if not os.path.exists(truth_path):
                truth_url = f"{download_root}/{truth_file}"
                urllib.request.urlretrieve(truth_url, truth_path)

# ------------------- Part IA: Dataset Parsing -------------------
def parse_data(ids_file: str, texts_file: str) -> List[Tuple[str, str]]:
    """Parse input files and process text according to requirements."""
    with open(ids_file, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    
    with open(texts_file, 'r') as f:
        texts = [re.sub(r'\W+', '', line.strip().lower()) for line in f.readlines()]
    
    return list(zip(ids, texts))

# ------------------- Part IB: Document Shingles -------------------
def shingle_document(text: str, k: int) -> Set[int]:
    """Create k-shingles from document and hash them."""
    shingles = set()
    for i in range(len(text) - k + 1):
        shingle = text[i:i+k]
        hashed_shingle = binascii.crc32(shingle.encode('utf-8')) & 0xffffffff
        shingles.add(hashed_shingle)
    return shingles

# ------------------- Part IC: Jaccard Similarity -------------------
def jaccard(set1: Set[int], set2: Set[int]) -> float:
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# ------------------- Part IIA: MinHash Preparation -------------------
def invert_shingles(shingled_documents: List[Tuple[str, Set[int]]]) -> Tuple[List[Tuple[int, str]], List[str]]:
    """Create inverted index of shingles to documents."""
    inv_index = []
    docids = []
    
    for docid, shingles in shingled_documents:
        docids.append(docid)
        for shingle in shingles:
            inv_index.append((shingle, docid))
    
    inv_index.sort()
    return inv_index, docids

# ------------------- Part IIB: Hash Functions -------------------
def make_random_hash_fn(p: int = 2**33-355, m: int = 4294967295):
    """Generate a random hash function."""
    a = np.random.randint(1, p-1)
    b = np.random.randint(0, p-1)
    return lambda x: ((a * x + b) % p) % m

def make_hashes(num_hashes: int) -> List:
    """Generate list of hash functions."""
    return [make_random_hash_fn() for _ in range(num_hashes)]

# ------------------- Part IIC: MinHash Signature Matrix -------------------
def make_minhash_signature(shingled_data: List[Tuple[str, Set[int]]], num_hashes: int) -> Tuple[np.ndarray, List[str]]:
    """Construct MinHash signature matrix."""
    inv_index, docids = invert_shingles(shingled_data)
    num_docs = len(docids)
    
    sigmatrix = np.full((num_hashes, num_docs), np.inf)
    hash_funcs = make_hashes(num_hashes)
    
    docid_to_idx = {docid: idx for idx, docid in enumerate(docids)}
    
    for shingle, docid in inv_index:
        doc_idx = docid_to_idx[docid]
        for i, hash_fn in enumerate(hash_funcs):
            hash_val = hash_fn(shingle)
            sigmatrix[i, doc_idx] = min(sigmatrix[i, doc_idx], hash_val)
    
    return sigmatrix, docids

# ------------------- Part IID: MinHash Similarity -------------------
def minhash_similarity(id1: str, id2: str, minhash_sigmat: np.ndarray, docids: List[str]) -> float:
    """Compute MinHash-based similarity estimate."""
    idx1 = docids.index(id1)
    idx2 = docids.index(id2)
    return np.mean(minhash_sigmat[:, idx1] == minhash_sigmat[:, idx2])

# ------------------- Part III: LSH Implementation -------------------
def choose_nbands(threshold: float, n: int) -> Tuple[int, float]:
    """Choose number of bands for LSH."""
    def error_fun(x):
        cur_t = (1/x[0])**(x[0]/n)
        return (threshold-cur_t)**2
    
    from scipy.optimize import minimize
    res = minimize(error_fun, x0=np.array([10]), method='Nelder-Mead')
    b = int(np.ceil(res.x[0]))
    r = int(n / b)
    return b, r

def do_lsh(minhash_sigmatrix: np.ndarray, numhashes: int, docids: List[str], threshold: float) -> List[Dict]:
    """Implement LSH using bands technique."""
    b, r = choose_nbands(threshold, numhashes)
    buckets = []
    
    for band in range(b):
        start_idx = band * r
        end_idx = min(start_idx + r, numhashes)
        
        cur_buckets = defaultdict(list)
        band_vectors = minhash_sigmatrix[start_idx:end_idx, :]
        
        for doc_idx, doc_id in enumerate(docids):
            vector = tuple(band_vectors[:, doc_idx])
            cur_buckets[vector].append(doc_id)
        
        buckets.append(cur_buckets)
    
    return buckets

def get_lsh_candidates(buckets: List[Dict]) -> List[Tuple[str, str]]:
    """Get candidate pairs from LSH buckets."""
    candidates = set()
    for bucket in buckets:
        for doc_ids in bucket.values():
            if len(doc_ids) > 1:
                for i in range(len(doc_ids)):
                    for j in range(i + 1, len(doc_ids)):
                        candidates.add(tuple(sorted([doc_ids[i], doc_ids[j]])))
    return list(candidates)

# ------------------- Evaluation and Visualization -------------------
def find_top_5_similar(doc_id: str, candidates: List[Tuple[str, str]], 
                      minhash_sigmat: np.ndarray, docids: List[str]) -> List[str]:
    """Find top 5 most similar documents for given document."""
    similarities = []
    for id1, id2 in candidates:
        if doc_id in (id1, id2):
            other_id = id2 if doc_id == id1 else id1
            similarity = minhash_similarity(doc_id, other_id, minhash_sigmat, docids)
            similarities.append((other_id, similarity))
    
    return [x[0] for x in sorted(similarities, key=lambda x: x[1], reverse=True)[:5]]

def evaluate_model(predictions: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> List[int]:
    """Evaluate model predictions against ground truth."""
    intersection_scores = []
    for doc_id in predictions:
        predicted_set = set(predictions[doc_id])
        truth_set = set(ground_truth.get(doc_id, []))
        intersection_scores.append(len(predicted_set.intersection(truth_set)))
    return intersection_scores

def plot_statistics(scores: List[int]):
    """Plot evaluation statistics."""
    df = pd.DataFrame({'scores': scores})
    print("\nStatistical Description:")
    print(df.describe())
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    sns.histplot(data=df, x='scores', bins=6)
    plt.title('Distribution of Intersection Scores')
    
    plt.subplot(122)
    sns.boxplot(x=df['scores'])
    plt.title('Box Plot of Intersection Scores')
    
    plt.tight_layout()
    plt.show()

# ------------------- Main Execution -------------------
def main():
    # Parameters
    k = 5  # shingle length
    num_hashes = 100  # number of hash functions
    threshold = 0.5  # LSH threshold
    
    # Load and process data
    ids_file = 'ids.txt'
    texts_file = 'texts.txt'
    ground_truth_file = 'items.json'
    
    # Parse input data
    parsed_data = parse_data(ids_file, texts_file)
    
    # Create shingles
    shingled_data = [(id_, shingle_document(text, k)) for id_, text in parsed_data]
    
    # Generate MinHash signatures
    minhash_sigmatrix, docids = make_minhash_signature(shingled_data, num_hashes)
    
    # Perform LSH
    buckets = do_lsh(minhash_sigmatrix, num_hashes, docids, threshold)
    candidates = get_lsh_candidates(buckets)
    
    # Find top 5 similar documents for each document
    predictions = {}
    for doc_id in docids:
        predictions[doc_id] = find_top_5_similar(doc_id, candidates, minhash_sigmatrix, docids)
    
    # Load ground truth and evaluate
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    intersection_scores = evaluate_model(predictions, ground_truth)
    
    # Plot results
    plot_statistics(intersection_scores)
    
    return predictions

if __name__ == "__main__":
    predictions = main()