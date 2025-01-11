import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Set
import pickle
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import faiss
import threading

class SemanticLSH:
    def __init__(self, num_hash_tables=10, num_hash_functions=8, device_id=1):
        """
        Initialize the SemanticLSH with specified parameters and FAISS index
        """
        self.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        # Using a smaller, faster model while maintaining good performance
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=self.device)
        self.num_hash_tables = num_hash_tables
        self.num_hash_functions = num_hash_functions
        self.hash_tables = None
        self.embeddings = None
        self.ids = None
        self.batch_size = 64  # Increased batch size for faster processing
        self.index = None
        self._lock = threading.Lock()
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using parallel processing and larger batches
        """
        def process_batch(batch):
            with torch.no_grad():
                return self.model.encode(batch, convert_to_tensor=True).cpu().numpy()

        # Process in larger batches
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            embeddings = list(tqdm(
                executor.map(process_batch, batches),
                total=len(batches),
                desc="Generating embeddings"
            ))
        
        return np.vstack(embeddings)

    def _build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS index for fast similarity search
        """
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def _hash_embeddings(self, embeddings: np.ndarray) -> List[np.ndarray]:
        """
        Optimized LSH signature generation using vectorized operations
        """
        dim = embeddings.shape[1]
        hash_tables = []
        
        # Generate all random vectors at once
        all_random_vectors = np.random.randn(self.num_hash_tables, self.num_hash_functions, dim)
        
        # Vectorized hash computation
        for random_vectors in all_random_vectors:
            projections = embeddings.dot(random_vectors.T)
            signatures = (projections > 0).astype(np.int8)  # Using int8 to save memory
            hash_tables.append(signatures)
            
        return hash_tables

    def fit(self, ids: List[str], texts: List[str]):
        """
        Fit the model with training data using optimized processing
        """
        self.ids = ids
        self.embeddings = self.generate_embeddings(texts)
        self.hash_tables = self._hash_embeddings(self.embeddings)
        self._build_faiss_index(self.embeddings.copy())  # Build FAISS index
        
    def get_similar_items(self, query_id: str, top_k: int = 5) -> List[str]:
        """
        Find similar items using FAISS for faster similarity search
        """
        with self._lock:  # Thread safety for concurrent queries
            query_idx = self.ids.index(query_id)
            query_embedding = self.embeddings[query_idx].reshape(1, -1)
            
            # Normalize query for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Use FAISS for fast similarity search
            D, I = self.index.search(query_embedding, top_k + 1)  # +1 to account for self
            
            # Remove self from results
            mask = I[0] != query_idx
            similar_indices = I[0][mask][:top_k]
            
            return [self.ids[idx] for idx in similar_indices]

    def save_model(self, path: str):
        """
        Save model state efficiently
        """
        state = {
            'hash_tables': [ht.astype(np.int8) for ht in self.hash_tables],  # Compress hash tables
            'embeddings': self.embeddings,
            'ids': self.ids
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=4)  # Use faster protocol

    def load_model(self, path: str):
        """
        Load model state and rebuild index
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)  # Corrected to load the state
        self.hash_tables = state['hash_tables']
        self.embeddings = state['embeddings']
        self.ids = state['ids']
        self._build_faiss_index(self.embeddings.copy())



def evaluate_model(model: SemanticLSH, ground_truth: Dict, plot_path: str = 'evaluation_plots.png'):
    """
    Parallel evaluation of model performance
    """
    def process_query(query_id):
        predicted = set(model.get_similar_items(query_id))
        actual = set(ground_truth[query_id])
        return len(predicted.intersection(actual))

    with ThreadPoolExecutor() as executor:
        intersection_scores = list(tqdm(
            executor.map(process_query, ground_truth.keys()),
            total=len(ground_truth),
            desc="Evaluating"
        ))
    
    # Calculate statistics
    scores_df = pd.DataFrame(intersection_scores, columns=['Intersection Score'])
    stats = scores_df.describe()
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(data=intersection_scores, bins=6, ax=ax1)
    ax1.set_title('Distribution of Intersection Scores')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Count')
    
    sns.boxplot(data=intersection_scores, ax=ax2)
    ax2.set_title('Box Plot of Intersection Scores')
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    return np.mean(intersection_scores), stats

with open('items.json', 'r') as f:
    ground_truth = json.load(f)
    
model = SemanticLSH()

model_file = 'semantic_lsh_model.pkl'
print("Loading existing model...")
model.load_model(model_file)

# Evaluate model
mean_score, stats = evaluate_model(model, ground_truth)
print(f"\nModel Performance:")
print(f"Mean Intersection Score: {mean_score:.2f}")
print("\nScore Statistics:")
print(stats)
