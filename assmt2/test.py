import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import os
import pickle
from datetime import datetime
import h5py

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

# class GPUEmbeddingProcessor:
#     def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         self.device = device
#         self.tfidf_vectorizer = None
#         self.glove_model = None
#         self.word2vec_model = None
#         print(f"Using device: {self.device}")
    
#     def save_models(self, output_dir):
#         """Save all models and vectorizers"""
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Save TF-IDF vectorizer
#         if self.tfidf_vectorizer:
#             with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
#                 pickle.dump(self.tfidf_vectorizer, f)
        
#         # Save GloVe model
#         if self.glove_model:
#             with open(os.path.join(output_dir, 'glove_model.pkl'), 'wb') as f:
#                 pickle.dump(self.glove_model, f)
    
#     def load_models(self, model_dir):
#         """Load all saved models and vectorizers"""
#         # Load TF-IDF vectorizer
#         tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
#         if os.path.exists(tfidf_path):
#             with open(tfidf_path, 'rb') as f:
#                 self.tfidf_vectorizer = pickle.load(f)
        
#         # Load GloVe model
#         glove_path = os.path.join(model_dir, 'glove_model.pkl')
#         if os.path.exists(glove_path):
#             with open(glove_path, 'rb') as f:
#                 self.glove_model = pickle.load(f)
    
#     def save_embeddings(self, combined_vectors, output_file):
#         """Save embeddings to HDF5 format"""
#         with h5py.File(output_file, 'w') as f:
#             f.create_dataset('embeddings', data=combined_vectors.cpu().numpy())
    
#     def load_embeddings(self, input_file):
#         """Load embeddings from HDF5 format"""
#         with h5py.File(input_file, 'r') as f:
#             embeddings = torch.tensor(f['embeddings'][:], device=self.device)
#         return embeddings
    
#     def process_batch(self, batch_vectors):
#         return torch.tensor(batch_vectors, device=self.device, dtype=torch.float32)
    
#     @torch.no_grad()
#     def compute_similarities(self, vectors, batch_size=128):
#         n_samples = len(vectors)
#         similarities = torch.zeros((n_samples, n_samples), device=self.device)
        
#         vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        
#         for i in tqdm(range(0, n_samples, batch_size)):
#             batch_end = min(i + batch_size, n_samples)
#             batch = vectors[i:batch_end]
#             similarities[i:batch_end] = torch.mm(batch, vectors.t())
        
#         return similarities

def load_glove_embeddings(file_path, device):
    print("Loading GloVe embeddings...")
    glove_embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(x) for x in values[1:]], device=device)
            glove_embeddings[word] = vector
    return glove_embeddings

def get_glove_sentence_embedding(sentence, glove_model, dim=300, device='cuda'):
    words = sentence.lower().split()
    embeddings = [glove_model.get(word, torch.zeros(dim, device=device)) for word in words]
    if embeddings:
        return torch.stack(embeddings).mean(dim=0)
    return torch.zeros(dim, device=device)

def save_results(output_dir, predicted_map, intersection_scores, intersection_df, combined_vectors):
    """Save all results and metrics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f'results_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save predictions
    with open(os.path.join(results_dir, 'predicted_items.json'), 'w') as f:
        json.dump(predicted_map, f, indent=4)
    
    # Save scores
    np.save(os.path.join(results_dir, 'intersection_scores.npy'), intersection_scores)
    
    # Save statistics
    stats_file = os.path.join(results_dir, 'statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("Intersection Score Statistics:\n")
        f.write(str(intersection_df.describe()))
    
    # Save plots
    plt.figure(figsize=(10, 6))
    plt.hist(intersection_scores, bins=6, edgecolor='black')
    plt.title('Histogram of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(results_dir, 'histogram.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(intersection_scores)
    plt.title('Box Plot of Intersection Scores')
    plt.ylabel('Intersection Score')
    plt.savefig(os.path.join(results_dir, 'boxplot.png'))
    plt.close()
    
    # Save embeddings
    embeddings_file = os.path.join(results_dir, 'embeddings.h5')
    with h5py.File(embeddings_file, 'w') as f:
        f.create_dataset('embeddings', data=combined_vectors.cpu().numpy())
    
    return results_dir

# def main():
#     # Configuration
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     batch_size = 1028
#     output_dir = 'model_output'
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load data
#     print("Loading data...")
#     with open('ids.txt', 'r') as f:
#         ids = f.read().splitlines()
#     with open('texts.txt', 'r') as f:
#         texts = f.read().splitlines()
#     with open('items.json', 'r') as f:
#         ground_truth = json.load(f)
    
#     # Initialize processor
#     processor = GPUEmbeddingProcessor(device)
    
#     # TF-IDF Embeddings
#     print("Computing TF-IDF embeddings...")
#     processor.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
#     tfidf_vectors = processor.tfidf_vectorizer.fit_transform(texts)
#     tfidf_tensor = torch.tensor(tfidf_vectors.toarray(), device=device, dtype=torch.float32)
    
#     # GloVe Embeddings
#     print("Computing GloVe embeddings...")
#     processor.glove_model = load_glove_embeddings('glove.6B.300d.txt', device)
#     glove_vectors = torch.stack([
#         get_glove_sentence_embedding(text, processor.glove_model, device=device)
#         for text in tqdm(texts)
#     ])
    
#     # Word2Vec Embeddings
#     print("Computing Word2Vec embeddings...")
#     word2vec_model = KeyedVectors.load_word2vec_format(
#         'GoogleNews-vectors-negative300.bin',
#         binary=True
#     )
    
#     word2vec_vectors = []
#     dataset = TextDataset(texts)
#     dataloader = DataLoader(dataset, batch_size=batch_size)
    
#     for batch in tqdm(dataloader):
#         batch_embeddings = []
#         for text in batch:
#             words = text.split()
#             word_vectors = [
#                 torch.tensor(word2vec_model[word], device=device)
#                 if word in word2vec_model
#                 else torch.zeros(300, device=device)
#                 for word in words
#             ]
#             if word_vectors:
#                 embedding = torch.stack(word_vectors).mean(dim=0)
#             else:
#                 embedding = torch.zeros(300, device=device)
#             batch_embeddings.append(embedding)
#         word2vec_vectors.extend(batch_embeddings)
    
#     word2vec_vectors = torch.stack(word2vec_vectors)
    
#     # Combine embeddings
#     print("Combining embeddings...")
#     combined_vectors = torch.cat([tfidf_tensor, glove_vectors, word2vec_vectors], dim=1)
    
#     # Save models and embeddings
#     print("Saving models and embeddings...")
#     processor.save_models(output_dir)
#     processor.save_embeddings(combined_vectors, os.path.join(output_dir, 'combined_embeddings.h5'))
    
#     # Compute similarities
#     print("Computing similarities...")
#     similarities = processor.compute_similarities(combined_vectors, batch_size)
    
#     # Get top 5 similar items
#     print("Finding top similar items...")
#     predicted_map = defaultdict(list)
#     topk_values, topk_indices = torch.topk(similarities, k=6, dim=1)
    
#     for idx, indices_tensor in enumerate(topk_indices):
#         indices = indices_tensor.cpu().numpy()
#         predicted_indices = [ids[i] for i in indices if i != idx][:5]
#         predicted_map[ids[idx]] = predicted_indices
    
#     # Evaluate
#     print("Evaluating results...")
#     intersection_scores = []
#     for sample_id in ids:
#         predicted_set = set(predicted_map[sample_id])
#         true_set = set(ground_truth.get(sample_id, []))
#         intersection_score = len(predicted_set.intersection(true_set))
#         intersection_scores.append(intersection_score)
    
#     # Create statistics
#     intersection_df = pd.DataFrame(intersection_scores, columns=['Intersection Score'])
#     print("\nStatistics:")
#     print(intersection_df.describe())
    
#     # Save all results
#     print("Saving results...")
#     results_dir = save_results(output_dir, predicted_map, intersection_scores, 
#                              intersection_df, combined_vectors)
#     print(f"Results saved to: {results_dir}")

def load_and_predict(text, model_dir='model_output'):
    """Function to load saved models and make predictions for new text"""
    processor = GPUEmbeddingProcessor()
    processor.load_models(model_dir)
    
    # Load saved embeddings
    combined_vectors = processor.load_embeddings(os.path.join(model_dir, 'combined_embeddings.h5'))
    
    # Process new text
    # (Add implementation for processing new text using loaded models)
    # This would use the same pipeline as above but for a single text input
    
    return combined_vectors

# if __name__ == "__main__":
#     main()
    
    
    
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import os
import pickle
from datetime import datetime
import h5py

# class GPUEmbeddingProcessor:
#     def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', chunk_size=1000):
#         self.device = device
#         self.chunk_size = chunk_size
#         self.tfidf_vectorizer = None
#         self.glove_model = None
#         print(f"Using device: {self.device}")
        
#         # Set memory-efficient CUDA allocation
#         if device == 'cuda':
#             torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of available GPU memory
#             torch.cuda.empty_cache()
    
#     @torch.no_grad()
#     def compute_similarities_chunked(self, vectors, chunk_size=None):
#         if chunk_size is None:
#             chunk_size = self.chunk_size
            
#         n_samples = len(vectors)
#         similarities = torch.zeros((n_samples, n_samples), device='cpu')
#         vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        
#         for i in tqdm(range(0, n_samples, chunk_size)):
#             chunk_end = min(i + chunk_size, n_samples)
#             chunk1 = vectors[i:chunk_end]
            
#             for j in range(0, n_samples, chunk_size):
#                 j_end = min(j + chunk_size, n_samples)
#                 chunk2 = vectors[j:j_end]
                
#                 # Compute similarity for current chunks
#                 chunk_sim = torch.mm(chunk1, chunk2.t())
#                 similarities[i:chunk_end, j:j_end] = chunk_sim.cpu()
                
#                 # Clear CUDA cache after each chunk
#                 if self.device == 'cuda':
#                     torch.cuda.empty_cache()
        
#         return similarities

#     def process_embeddings_in_chunks(self, texts, batch_size=32):
#         # Process TF-IDF
#         print("Computing TF-IDF embeddings...")
#         if self.tfidf_vectorizer is None:
#             self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
#             self.tfidf_vectorizer.fit(texts)
        
#         all_embeddings = []
#         dataset = TextDataset(texts)
#         dataloader = DataLoader(dataset, batch_size=batch_size)
        
#         for batch in tqdm(dataloader):
#             # TF-IDF
#             tfidf_vectors = self.tfidf_vectorizer.transform(batch).toarray()
#             tfidf_tensor = torch.tensor(tfidf_vectors, device=self.device, dtype=torch.float32)
            
#             # GloVe
#             glove_vectors = torch.stack([
#                 get_glove_sentence_embedding(text, self.glove_model, device=self.device)
#                 for text in batch
#             ])
            
#             # Combine embeddings for this batch
#             combined = torch.cat([tfidf_tensor, glove_vectors], dim=1)
#             all_embeddings.append(combined.cpu())
            
#             if self.device == 'cuda':
#                 torch.cuda.empty_cache()
        
#         return torch.cat(all_embeddings, dim=0).to(self.device)






"""Try2"""
# class GPUEmbeddingProcessor:
#     def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', chunk_size=1000):
#         self.device = device
#         self.chunk_size = chunk_size
#         self.tfidf_vectorizer = None
#         self.glove_model = None
#         print(f"Using device: {self.device}")
        
#         # Set memory-efficient CUDA allocation
#         if device == 'cuda':
#             torch.cuda.set_per_process_memory_fraction(0.7)
#             torch.cuda.empty_cache()
    
#     @torch.no_grad()
#     def compute_similarities_chunked(self, vectors, chunk_size=None):
#         if chunk_size is None:
#             chunk_size = self.chunk_size
            
#         n_samples = len(vectors)
#         similarities = torch.zeros((n_samples, n_samples), device='cpu')
#         vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        
#         for i in tqdm(range(0, n_samples, chunk_size)):
#             chunk_end = min(i + chunk_size, n_samples)
#             chunk1 = vectors[i:chunk_end]
            
#             for j in range(0, n_samples, chunk_size):
#                 j_end = min(j + chunk_size, n_samples)
#                 chunk2 = vectors[j:j_end]
                
#                 chunk_sim = torch.mm(chunk1, chunk2.t())
#                 similarities[i:chunk_end, j:j_end] = chunk_sim.cpu()
                
#                 if self.device == 'cuda':
#                     torch.cuda.empty_cache()
        
#         return similarities

#     def process_embeddings_in_chunks(self, texts, batch_size=32):
#         print("Computing TF-IDF embeddings...")
#         if self.tfidf_vectorizer is None:
#             self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
#             self.tfidf_vectorizer.fit(texts)
        
#         all_embeddings = []
#         dataset = TextDataset(texts)
#         dataloader = DataLoader(dataset, batch_size=batch_size)
        
#         for batch in tqdm(dataloader):
#             tfidf_vectors = self.tfidf_vectorizer.transform(batch).toarray()
#             tfidf_tensor = torch.tensor(tfidf_vectors, device=self.device, dtype=torch.float32)
            
#             glove_vectors = torch.stack([
#                 get_glove_sentence_embedding(text, self.glove_model, device=self.device)
#                 for text in batch
#             ])
            
#             combined = torch.cat([tfidf_tensor, glove_vectors], dim=1)
#             all_embeddings.append(combined.cpu())
            
#             if self.device == 'cuda':
#                 torch.cuda.empty_cache()
        
#         return torch.cat(all_embeddings, dim=0).to(self.device)

# def get_top_similar_items(similarities_row, current_idx, ids, k=6):
#     """
#     Get top similar items with proper index validation
#     """
#     n_samples = len(similarities_row)
#     # Create mask to exclude the current index
#     mask = torch.ones(n_samples, dtype=torch.bool)
#     mask[current_idx] = False
    
#     # Apply mask and get top k
#     masked_similarities = similarities_row[mask]
#     top_k_values, top_k_indices = torch.topk(masked_similarities, min(k, len(masked_similarities)))
    
#     # Convert masked indices back to original indices
#     original_indices = torch.arange(n_samples)[mask][top_k_indices]
    
#     # Ensure indices are within bounds
#     valid_indices = [idx.item() for idx in original_indices if idx < len(ids)]
#     return [ids[idx] for idx in valid_indices][:5]

# def main():
#     # Configuration
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     batch_size = 32
#     chunk_size = 1000
#     output_dir = 'model_output'
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load data
#     print("Loading data...")
#     with open('ids.txt', 'r') as f:
#         ids = [line.strip() for line in f.readlines()]

#     # Read texts
#     with open('texts.txt', 'r') as f:
#         texts = [line.strip() for line in f.readlines()]

#     # Load ground truth items from JSON
#     with open('items.json', 'r') as f:
#         ground_truth = json.load(f)
    
#     # Validate data
#     print(f"Number of ids: {len(ids)}")
#     print(f"Number of texts: {len(texts)}")
#     assert len(ids) == len(texts), "Number of ids and texts must match"
    
#     # Initialize processor
#     processor = GPUEmbeddingProcessor(device, chunk_size=chunk_size)
    
#     # Load GloVe embeddings
#     print("Loading GloVe embeddings...")
#     processor.glove_model = load_glove_embeddings('glove.6B.300d.txt', device)
    
#     # Process embeddings in chunks
#     combined_vectors = processor.process_embeddings_in_chunks(texts, batch_size)
    
#     # Compute similarities in chunks
#     print("Computing similarities...")
#     similarities = processor.compute_similarities_chunked(combined_vectors)
    
#     # Get top 5 similar items
#     print("Finding top similar items...")
#     predicted_map = defaultdict(list)
#     for i in tqdm(range(len(ids))):
#         similarities_row = similarities[i]
#         predicted_indices = get_top_similar_items(similarities_row, i, ids)
#         predicted_map[ids[i]] = predicted_indices
        
#         # Validate predictions
#         if len(predicted_indices) < 5:
#             print(f"Warning: Only found {len(predicted_indices)} similar items for id {ids[i]}")
    
#     # Evaluate
#     print("Evaluating results...")
#     intersection_scores = []
#     for sample_id in ids:
#         predicted_set = set(predicted_map[sample_id])
#         true_set = set(ground_truth.get(sample_id, []))
#         intersection_score = len(predicted_set.intersection(true_set))
#         intersection_scores.append(intersection_score)
    
#     # Create statistics
#     intersection_df = pd.DataFrame(intersection_scores, columns=['Intersection Score'])
#     print("\nStatistics:")
#     print(intersection_df.describe())
    
#     # Save results
#     print("Saving results...")
#     results_dir = save_results(output_dir, predicted_map, intersection_scores, 
#                              intersection_df, combined_vectors)
#     print(f"Results saved to: {results_dir}")

# if __name__ == "__main__":
#     main()




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
            state = pickle.dump(f)
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

def main():
    # Enable tensor cores for faster GPU computation
    torch.cuda.set_device(1)

    # Set to GPU device ID 1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load data efficiently
    print("Loading data...")
    with open('ids.txt', 'r') as f:
        ids = f.read().splitlines()  # Faster than readlines()
    
    with open('texts.txt', 'r') as f:
        texts = f.read().splitlines()
    
    with open('items.json', 'r') as f:
        ground_truth = json.load(f)
    
    # Initialize and train model
    model = SemanticLSH(num_hash_tables=10, num_hash_functions=8, device_id=1)
    model.fit(ids, texts)
    
    # Save model
    model.save_model('semantic_lsh_model.pkl')
    
    # Evaluate model
    mean_score, stats = evaluate_model(model, ground_truth)
    print(f"\nModel Performance:")
    print(f"Mean Intersection Score: {mean_score:.2f}")
    print("\nScore Statistics:")
    print(stats)

if __name__ == "__main__":
    main()