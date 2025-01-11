import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
# Step 1: Load the Data
def load_data():
    # Load ids.txt
    with open('ids.txt', 'r') as f:
        ids = [line.strip() for line in f.readlines()]

    # Load texts.txt
    with open('texts.txt', 'r') as f:
        texts = [line.strip() for line in f.readlines()]

    # Load items.json (Ground Truth)
    with open('items.json', 'r') as f:
        items = json.load(f)
        
    return ids, texts, items
# Step 2: Text Vectorization using TF-IDF
def vectorize_texts(texts):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    text_vectors = vectorizer.fit_transform(texts)
    return text_vectors
# Step 3: Implement LSH using MinHash
def compute_minhash_lsh(texts, num_perm=128):
    lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
    minhashes = []
    
    for i, text in enumerate(texts):
        minhash = MinHash(num_perm=num_perm)
        for word in text.split():
            minhash.update(word.encode('utf8'))
        lsh.insert(f"id_{i}", minhash)
        minhashes.append(minhash)
    
    return lsh, minhashes

# Step 4: Predict Top 5 Similar Items using LSH
def get_top_5_similar(lsh, minhashes, ids):
    top_5_predictions = {}
    
    for i, minhash in enumerate(minhashes):
        similar_items = lsh.query(minhash)  # Query similar items in LSH
        similar_items = [item for item in similar_items if item != f"id_{i}"]  # Exclude self
        top_5 = similar_items[:5]  # Get the top 5 similar items
        top_5_ids = [ids[int(item.split('_')[1])] for item in top_5]
        top_5_predictions[ids[i]] = top_5_ids
        
    return top_5_predictions
# Step 5: Evaluate Intersection Score with Ground Truth
def evaluate(predictions, ground_truth):
    intersection_scores = []
    
    for sample_id in predictions:
        pred_items = set(predictions[sample_id])
        true_items = set(ground_truth.get(sample_id, []))
        intersection_score = len(pred_items.intersection(true_items))
        intersection_scores.append(intersection_score)
        
    avg_score = np.mean(intersection_scores)
    return intersection_scores, avg_score
# Step 6: Visualizations - Histogram, Box Plot, and Statistics
def plot_statistics(intersection_scores):
    scores_series = pd.Series(intersection_scores)
    
    # Descriptive statistics
    print(scores_series.describe())

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(intersection_scores, bins=range(0, 6), alpha=0.7, edgecolor='black')
    plt.title('Histogram of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.ylabel('Frequency')
    plt.show()

    # Box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(intersection_scores, vert=False)
    plt.title('Box Plot of Intersection Scores')
    plt.show()
# Text Preprocessing Function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'\W+', ' ', text)
    
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Stemming (use lemmatization if preferred)
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    
    return ' '.join(words)
# Step 7: Main Function

# Load data
ids, texts, ground_truth = load_data()

# Vectorize texts
text_vectors = vectorize_texts(texts)

# Compute LSH
lsh, minhashes = compute_minhash_lsh(texts)

# Get top 5 similar items for each sample
predictions = get_top_5_similar(lsh, minhashes, ids)

# Evaluate the model
intersection_scores, avg_score = evaluate(predictions, ground_truth)
print(f"Average Intersection Score: {avg_score:.2f}")

# Visualize results
plot_statistics(intersection_scores)
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Step 1: Load the Data
def load_data():
    # Load ids.txt
    with open('ids.txt', 'r') as f:
        ids = [line.strip() for line in f.readlines()]

    # Load texts.txt
    with open('texts.txt', 'r') as f:
        texts = [line.strip() for line in f.readlines()]

    # Load items.json (Ground Truth)
    with open('items.json', 'r') as f:
        items = json.load(f)
        
    return ids, texts, items

# Text Preprocessing Function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and special characters (keep apostrophes for contractions like "don't")
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Remove numeric tokens (optional)
    words = [word for word in words if not word.isdigit()]
    
    # Lemmatization (use lemmatization instead of stemming for better results)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Step 2: Text Vectorization using TF-IDF with Feature Engineering
def vectorize_texts(texts, ngram_range=(1, 2), max_features=1000):
    # Preprocess the texts before vectorizing
    texts = [preprocess_text(text) for text in texts]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=ngram_range)
    text_vectors = vectorizer.fit_transform(texts)
    
    return text_vectors

# Step 3: Implement LSH using MinHash
def compute_minhash_lsh(texts, num_perm=128):
    lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
    minhashes = []
    
    for i, text in enumerate(texts):
        minhash = MinHash(num_perm=num_perm)
        for word in text.split():
            minhash.update(word.encode('utf8'))
        lsh.insert(f"id_{i}", minhash)
        minhashes.append(minhash)
    
    return lsh, minhashes

# Step 4: Predict Top 5 Similar Items using LSH
def get_top_5_similar(lsh, minhashes, ids):
    top_5_predictions = {}
    
    for i, minhash in enumerate(minhashes):
        similar_items = lsh.query(minhash)  # Query similar items in LSH
        similar_items = [item for item in similar_items if item != f"id_{i}"]  # Exclude self
        top_5 = similar_items[:5]  # Get the top 5 similar items
        top_5_ids = [ids[int(item.split('_')[1])] for item in top_5]
        top_5_predictions[ids[i]] = top_5_ids
        
    return top_5_predictions

# Step 5: Evaluate Intersection Score with Ground Truth
def evaluate(predictions, ground_truth):
    intersection_scores = []
    
    for sample_id in predictions:
        pred_items = set(predictions[sample_id])
        true_items = set(ground_truth.get(sample_id, []))
        intersection_score = len(pred_items.intersection(true_items))
        intersection_scores.append(intersection_score)
        
    avg_score = np.mean(intersection_scores)
    return intersection_scores, avg_score

# Step 6: Visualizations - Histogram, Box Plot, and Statistics
def plot_statistics(intersection_scores):
    scores_series = pd.Series(intersection_scores)
    
    # Descriptive statistics
    print(scores_series.describe())

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(intersection_scores, bins=range(0, 6), alpha=0.7, edgecolor='black')
    plt.title('Histogram of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.ylabel('Frequency')
    plt.show()

    # Box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(intersection_scores, vert=False)
    plt.title('Box Plot of Intersection Scores')
    plt.show()

# Step 7: Main Function
def main():
    # Load data
    ids, texts, ground_truth = load_data()
    
    # Vectorize texts with Feature Engineering
    text_vectors = vectorize_texts(texts, ngram_range=(1, 2), max_features=2000)
    
    # Compute LSH
    lsh, minhashes = compute_minhash_lsh(texts)
    
    # Get top 5 similar items for each sample
    predictions = get_top_5_similar(lsh, minhashes, ids)
    
    # Evaluate the model
    intersection_scores, avg_score = evaluate(predictions, ground_truth)
    print(f"Average Intersection Score: {avg_score:.2f}")
    
    # Visualize results
    plot_statistics(intersection_scores)

# Run the main function
if __name__ == "__main__":
    main()

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH

# Step 0: Download required NLTK resources
def download_nltk_resources():
    nltk.download('punkt')       # For tokenization
    nltk.download('stopwords')   # For stopword removal
    nltk.download('wordnet')     # For lemmatization
    nltk.download('omw-1.4')     # For WordNet lemmatizer

# Step 1: Load the Data
def load_data():
    # Load ids.txt
    with open('ids.txt', 'r', encoding='utf-8') as f:
        ids = [line.strip() for line in f.readlines()]

    # Load texts.txt
    with open('texts.txt', 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]

    # Load items.json (Ground Truth)
    with open('items.json', 'r', encoding='utf-8') as f:
        items = json.load(f)
        
    return ids, texts, items

# Step 2: Text Preprocessing Function
def preprocess_text(text):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and special characters (retain apostrophes if needed)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Remove numeric tokens
    words = [word for word in words if not word.isdigit()]
    
    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Step 3: Text Vectorization using TF-IDF with Feature Engineering
def vectorize_texts(texts, ngram_range=(1, 2), max_features=2000, min_df=2, max_df=0.95):
    # Preprocess the texts before vectorizing
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )
    text_vectors = vectorizer.fit_transform(preprocessed_texts)
    
    return vectorizer, text_vectors

# Step 4: Implement LSH using MinHash
def compute_minhash_lsh(texts, num_perm=256, threshold=0.5):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []
    
    for i, text in enumerate(texts):
        minhash = MinHash(num_perm=num_perm)
        for word in text.split():
            minhash.update(word.encode('utf8'))
        lsh.insert(f"id_{i}", minhash)
        minhashes.append(minhash)
    
    return lsh, minhashes

# Step 5: Predict Top 5 Similar Items using LSH
def get_top_5_similar(lsh, minhashes, ids):
    top_5_predictions = {}
    
    for i, minhash in enumerate(minhashes):
        similar_items = lsh.query(minhash)  # Query similar items in LSH
        # Exclude self
        similar_items = [item for item in similar_items if item != f"id_{i}"]
        # If less than 5 similar items found, you may need to adjust threshold or handle accordingly
        top_5 = similar_items[:5]  # Get the top 5 similar items
        top_5_ids = [ids[int(item.split('_')[1])] for item in top_5]
        top_5_predictions[ids[i]] = top_5_ids
        
    return top_5_predictions

# Step 6: Evaluate Intersection Score with Ground Truth
def evaluate(predictions, ground_truth):
    intersection_scores = []
    
    for sample_id in predictions:
        pred_items = set(predictions[sample_id])
        true_items = set(ground_truth.get(sample_id, []))
        intersection_score = len(pred_items.intersection(true_items))
        intersection_scores.append(intersection_score)
        
    avg_score = np.mean(intersection_scores)
    return intersection_scores, avg_score

# Step 7: Visualizations - Histogram, Box Plot, and Statistics
def plot_statistics(intersection_scores):
    scores_series = pd.Series(intersection_scores)
    
    # Descriptive statistics
    print("Descriptive Statistics:")
    print(scores_series.describe())

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(intersection_scores, bins=range(0, 7), alpha=0.7, edgecolor='black', align='left')
    plt.title('Histogram of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.ylabel('Frequency')
    plt.xticks(range(0, 6))
    plt.show()

    # Box plot
    plt.figure(figsize=(10, 2))
    plt.boxplot(intersection_scores, vert=False)
    plt.title('Box Plot of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.show()

# Step 8: Save Model Function
def save_model(vectorizer, minhashes, lsh, filename_prefix="model"):
    os.makedirs('models', exist_ok=True)  # Create directory to store models
    with open(f'models/{filename_prefix}_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(f'models/{filename_prefix}_minhashes.pkl', 'wb') as f:
        pickle.dump(minhashes, f)
    
    with open(f'models/{filename_prefix}_lsh.pkl', 'wb') as f:
        pickle.dump(lsh, f)

# Step 9: Load Model Function
def load_model(filename_prefix="model"):
    with open(f'models/{filename_prefix}_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(f'models/{filename_prefix}_minhashes.pkl', 'rb') as f:
        minhashes = pickle.load(f)
    
    with open(f'models/{filename_prefix}_lsh.pkl', 'rb') as f:
        lsh = pickle.load(f)
    
    return vectorizer, minhashes, lsh

# Step 10: Main Function
def main():
    # Download NLTK resources
    download_nltk_resources()
    
    # Load data
    ids, texts, ground_truth = load_data()
    
    # Vectorize texts with Feature Engineering
    vectorizer, text_vectors = vectorize_texts(
        texts,
        ngram_range=(1, 2),
        max_features=2000,
        min_df=2,
        max_df=0.95
    )
    
    # Compute LSH
    lsh, minhashes = compute_minhash_lsh(texts, num_perm=256, threshold=0.5)
    
    # Save the model
    save_model(vectorizer, minhashes, lsh, filename_prefix="text_similarity_model")
    print("Model saved successfully.")
    
    # Get top 5 similar items for each sample
    predictions = get_top_5_similar(lsh, minhashes, ids)
    
    # Evaluate the model
    intersection_scores, avg_score = evaluate(predictions, ground_truth)
    print(f"Average Intersection Score: {avg_score:.2f}")
    
    # Visualize results
    plot_statistics(intersection_scores)
    
    # Optionally, save predictions to a JSON file
    with open('predictions.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4)
    print("Predictions saved to 'predictions.json'.")

# Run the main function
if __name__ == "__main__":
    main()

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datasketch import MinHash, MinHashLSH

# Step 0: Download required NLTK resources
def download_nltk_resources():
    nltk.download('punkt')       # For tokenization
    nltk.download('stopwords')   # For stopword removal
    nltk.download('wordnet')     # For lemmatization
    nltk.download('omw-1.4')     # For WordNet lemmatizer
    nltk.download('maxent_ne_chunker')  # For Named Entity Recognition
    nltk.download('words')  # For NER

# Step 1: Load the Data
def load_data():
    # Load ids.txt
    with open('ids.txt', 'r', encoding='utf-8') as f:
        ids = [line.strip() for line in f.readlines()]

    # Load texts.txt
    with open('texts.txt', 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]

    # Load items.json (Ground Truth)
    with open('items.json', 'r', encoding='utf-8') as f:
        items = json.load(f)
        
    return ids, texts, items

# Step 2: Text Preprocessing Function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and not word.isdigit()]
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Step 3: Extract Named Entities
def extract_entities(text):
    from nltk import pos_tag, ne_chunk
    
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tree = ne_chunk(pos_tags)
    
    entities = []
    for subtree in tree.subtrees():
        if subtree.label() in ['PERSON', 'ORGANIZATION', 'GPE']:  # You can add more entity types if needed
            entities.append(' '.join(word for word, _ in subtree.leaves()))
    
    return ' '.join(entities)

# Step 4: Vectorization and Topic Modeling
def vectorize_texts(texts, ngram_range, max_features, min_df, max_df):
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )
    text_vectors = vectorizer.fit_transform(texts)
    
    # Return only vectorizer and text_vectors
    return vectorizer, text_vectors


# Step 5: Dimensionality Reduction
def apply_dimensionality_reduction(text_vectors):
    # Standardizing the feature vectors
    scaler = StandardScaler(with_mean=False)
    scaled_vectors = scaler.fit_transform(text_vectors)

    # Applying PCA
    pca = PCA(n_components=2)  # Adjust number of components as needed
    reduced_vectors = pca.fit_transform(scaled_vectors)

    return reduced_vectors

# Step 6: Implement LSH using MinHash
def compute_minhash_lsh(texts, num_perm=256, threshold=0.5):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []
    
    for i, text in enumerate(texts):
        minhash = MinHash(num_perm=num_perm)
        for word in text.split():
            minhash.update(word.encode('utf8'))
        lsh.insert(f"id_{i}", minhash)
        minhashes.append(minhash)
    
    return lsh, minhashes

# Step 7: Predict Top 5 Similar Items using LSH
def get_top_5_similar(lsh, minhashes, ids):
    top_5_predictions = {}
    
    for i, minhash in enumerate(minhashes):
        similar_items = lsh.query(minhash)
        similar_items = [item for item in similar_items if item != f"id_{i}"]
        top_5 = similar_items[:5]
        top_5_ids = [ids[int(item.split('_')[1])] for item in top_5]
        top_5_predictions[ids[i]] = top_5_ids
        
    return top_5_predictions

# Step 8: Evaluate Intersection Score with Ground Truth
def evaluate(predictions, ground_truth):
    intersection_scores = []
    
    for sample_id in predictions:
        pred_items = set(predictions[sample_id])
        true_items = set(ground_truth.get(sample_id, []))
        intersection_score = len(pred_items.intersection(true_items))
        intersection_scores.append(intersection_score)
        
    avg_score = np.mean(intersection_scores)
    return intersection_scores, avg_score

# Step 9: Visualizations - Histogram, Box Plot, and Statistics
def plot_statistics(intersection_scores):
    scores_series = pd.Series(intersection_scores)
    
    print("Descriptive Statistics:")
    print(scores_series.describe())

    plt.figure(figsize=(10, 6))
    plt.hist(intersection_scores, bins=range(0, 7), alpha=0.7, edgecolor='black', align='left')
    plt.title('Histogram of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.ylabel('Frequency')
    plt.xticks(range(0, 6))
    plt.show()

    plt.figure(figsize=(10, 2))
    plt.boxplot(intersection_scores, vert=False)
    plt.title('Box Plot of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.show()

# Step 10: Save Model Function
def save_model(vectorizer, minhashes, lsh, filename_prefix="model"):
    os.makedirs('models', exist_ok=True)
    with open(f'models/{filename_prefix}_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'models/{filename_prefix}_minhashes.pkl', 'wb') as f:
        pickle.dump(minhashes, f)
    with open(f'models/{filename_prefix}_lsh.pkl', 'wb') as f:
        pickle.dump(lsh, f)


# Step 11: Load Model Function
def load_model(filename_prefix="model"):
    with open(f'models/{filename_prefix}_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(f'models/{filename_prefix}_lda.pkl', 'rb') as f:
        lda = pickle.load(f)

    with open(f'models/{filename_prefix}_minhashes.pkl', 'rb') as f:
        minhashes = pickle.load(f)
    
    with open(f'models/{filename_prefix}_lsh.pkl', 'rb') as f:
        lsh = pickle.load(f)
    
    return vectorizer, lda, minhashes, lsh

from sklearn.decomposition import TruncatedSVD

def apply_dimensionality_reduction(text_vectors, n_components=2):
    # Applying Truncated SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=n_components)
    reduced_vectors = svd.fit_transform(text_vectors)
    
    return reduced_vectors


# Step 12: Main Function
def main():
    # Download NLTK resources
    download_nltk_resources()
    
    # Load data
    ids, texts, ground_truth = load_data()
    
    # Vectorize texts with Feature Engineering
    vectorizer, text_vectors = vectorize_texts(
        texts,
        ngram_range=(1, 2),
        max_features=2000,
        min_df=2,
        max_df=0.95
    )
    
    # Apply Dimensionality Reduction
    reduced_vectors = apply_dimensionality_reduction(text_vectors)
    
    # Compute LSH
    
    lsh, minhashes = compute_minhash_lsh(texts, num_perm=256, threshold=0.5)
    
    # Save the model
    save_model(vectorizer, minhashes, lsh, filename_prefix="text_similarity_model")
    print("Model saved successfully.")
    
    # Get top 5 similar items for each sample
    predictions = get_top_5_similar(lsh, minhashes, ids)
    
    # Evaluate the model
    intersection_scores, avg_score = evaluate(predictions, ground_truth)
    print(f"Average Intersection Score: {avg_score:.2f}")
    
    # Visualize results
    plot_statistics(intersection_scores)
    
    # Optionally, save predictions to a JSON file
    with open('predictions.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4)
    print("Predictions saved to 'predictions.json'.")


# Run the main function
if __name__ == "__main__":
    main()

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from datasketch import MinHash, MinHashLSH

# Step 0: Download required NLTK resources
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Step 1: Load the Data
def load_data():
    with open('ids.txt', 'r', encoding='utf-8') as f:
        ids = [line.strip() for line in f.readlines()]

    with open('texts.txt', 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]

    with open('items.json', 'r', encoding='utf-8') as f:
        items = json.load(f)

    return ids, texts, items

# Step 2: Text Preprocessing Function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()

    # Lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and not word.isdigit()]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

# Step 3: Vectorization with Feature Engineering (TF-IDF with bigrams)
def vectorize_texts(texts, ngram_range=(1, 2), max_features=2000, min_df=2, max_df=0.95):
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )
    text_vectors = vectorizer.fit_transform(texts)
    return vectorizer, text_vectors

# Step 4: Dimensionality Reduction using Truncated SVD
def apply_dimensionality_reduction(text_vectors, n_components=2):
    svd = TruncatedSVD(n_components=n_components)
    reduced_vectors = svd.fit_transform(text_vectors)
    return reduced_vectors

# Step 5: Implement LSH using MinHash
def compute_minhash_lsh(texts, num_perm=256, threshold=0.5):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []
    
    for i, text in enumerate(texts):
        minhash = MinHash(num_perm=num_perm)
        for word in text.split():
            minhash.update(word.encode('utf8'))
        lsh.insert(f"id_{i}", minhash)
        minhashes.append(minhash)
    
    return lsh, minhashes

# Step 6: Predict Top 5 Similar Items using LSH
def get_top_5_similar(lsh, minhashes, ids):
    top_5_predictions = {}
    
    for i, minhash in enumerate(minhashes):
        similar_items = lsh.query(minhash)
        similar_items = [item for item in similar_items if item != f"id_{i}"]
        top_5 = similar_items[:5]
        top_5_ids = [ids[int(item.split('_')[1])] for item in top_5]
        top_5_predictions[ids[i]] = top_5_ids
        
    return top_5_predictions

# Step 7: Evaluate Intersection Score with Ground Truth
def evaluate(predictions, ground_truth):
    intersection_scores = []
    
    for sample_id in predictions:
        pred_items = set(predictions[sample_id])
        true_items = set(ground_truth.get(sample_id, []))
        intersection_score = len(pred_items.intersection(true_items))
        intersection_scores.append(intersection_score)
        
    avg_score = np.mean(intersection_scores)
    return intersection_scores, avg_score

# Step 8: Visualizations - Histogram and Box Plot
def plot_statistics(intersection_scores):
    scores_series = pd.Series(intersection_scores)
    
    print("Descriptive Statistics:")
    print(scores_series.describe())

    plt.figure(figsize=(10, 6))
    plt.hist(intersection_scores, bins=range(0, 7), alpha=0.7, edgecolor='black', align='left')
    plt.title('Histogram of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.ylabel('Frequency')
    plt.xticks(range(0, 6))
    plt.show()

    plt.figure(figsize=(10, 2))
    plt.boxplot(intersection_scores, vert=False)
    plt.title('Box Plot of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.show()

# Step 9: Save the Model
def save_model(vectorizer, minhashes, lsh, filename_prefix="model"):
    os.makedirs('models', exist_ok=True)
    with open(f'models/{filename_prefix}_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'models/{filename_prefix}_minhashes.pkl', 'wb') as f:
        pickle.dump(minhashes, f)
    with open(f'models/{filename_prefix}_lsh.pkl', 'wb') as f:
        pickle.dump(lsh, f)

# Step 10: Main Function
def main():
    # Download NLTK resources
    download_nltk_resources()
    
    # Load data
    ids, texts, ground_truth = load_data()

    # Preprocess texts
    preprocessed_texts = [preprocess_text(text) for text in texts]

    # Feature engineering - Vectorization (TF-IDF with Bigrams)
    vectorizer, text_vectors = vectorize_texts(
        preprocessed_texts,
        ngram_range=(1, 2),
        max_features=2000,
        min_df=2,
        max_df=0.95
    )

    # Dimensionality Reduction
    reduced_vectors = apply_dimensionality_reduction(text_vectors)

    # Compute LSH
    lsh, minhashes = compute_minhash_lsh(preprocessed_texts)

    # Save the model
    save_model(vectorizer, minhashes, lsh, filename_prefix="text_similarity_model")
    print("Model saved successfully.")

    # Get top 5 similar items for each sample
    predictions = get_top_5_similar(lsh, minhashes, ids)

    # Evaluate the model
    intersection_scores, avg_score = evaluate(predictions, ground_truth)
    print(f"Average Intersection Score: {avg_score:.2f}")

    # Visualize results
    plot_statistics(intersection_scores)

    # Optionally, save predictions to a JSON file
    with open('predictions.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4)
    print("Predictions saved to 'predictions.json'.")

# Run the main function
if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from datasketch import MinHash, MinHashLSH
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading and Preprocessing
def load_data(ids_file, texts_file, items_file=None):
    with open(ids_file, 'r') as f:
        ids = [line.strip() for line in f]
    with open(texts_file, 'r') as f:
        texts = [line.strip() for line in f]
    
    data = pd.DataFrame({'id': ids, 'text': texts})
    
    if items_file:
        with open(items_file, 'r') as f:
            ground_truth = json.load(f)
        return data, ground_truth
    return data

# 2. Feature Engineering
def engineer_features(texts):
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Normalize the TF-IDF matrix
    normalized_matrix = normalize(tfidf_matrix, norm='l2', axis=1)
    
    return normalized_matrix, vectorizer

# 3. LSH Implementation
def create_minhash_lsh(matrix, num_perm=128):
    minhashes = []
    for vec in matrix:
        mh = MinHash(num_perm=num_perm)
        for idx in vec.nonzero()[1]:
            mh.update(str(idx).encode('utf-8'))
        minhashes.append(mh)
    
    lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
    for idx, mh in enumerate(minhashes):
        lsh.insert(idx, mh)
    
    return lsh, minhashes

# 4. Similarity Computation
def find_similar_items(lsh, minhashes, k=5):
    similar_items = {}
    for idx, mh in enumerate(minhashes):
        results = lsh.query(mh)
        similar = [r for r in results if r != idx]
        similar_items[idx] = similar[:k]
    return similar_items

# 5. Evaluation
def evaluate(predictions, ground_truth):
    intersection_scores = []
    for idx, pred in predictions.items():
        gt = ground_truth.get(str(idx), [])
        intersection = set(pred).intersection(set(gt))
        intersection_scores.append(len(intersection))
    return intersection_scores

# 6. Visualization
def visualize_results(intersection_scores):
    plt.figure(figsize=(12, 4))
    
    # Histogram
    plt.subplot(131)
    plt.hist(intersection_scores, bins=6, range=(0, 5))
    plt.title('Histogram of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.ylabel('Frequency')
    
    # Box Plot
    plt.subplot(132)
    plt.boxplot(intersection_scores)
    plt.title('Box Plot of Intersection Scores')
    plt.ylabel('Intersection Score')
    
    # Statistics
    plt.subplot(133)
    stats = pd.Series(intersection_scores).describe()
    plt.axis('off')
    plt.text(0.1, 0.9, str(stats), fontsize=10, verticalalignment='top')
    plt.title('Statistics of Intersection Scores')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    train_data, ground_truth = load_data('ids.txt', 'texts.txt', 'items.json')
    
    # Feature engineering
    matrix, vectorizer = engineer_features(train_data['text'])
    
    # Create LSH index
    lsh, minhashes = create_minhash_lsh(matrix)
    
    # Find similar items
    similar_items = find_similar_items(lsh, minhashes)
    
    # Evaluate
    intersection_scores = evaluate(similar_items, ground_truth)
    
    # Visualize results
    visualize_results(intersection_scores)
    
    # Calculate average score
    average_score = np.mean(intersection_scores)
    print(f"Average Intersection Score: {average_score:.2f}")
    
    # Function to predict for test data
    def predict_test(test_ids_file, test_texts_file):
        test_data = load_data(test_ids_file, test_texts_file)
        test_matrix = vectorizer.transform(test_data['text'])
        test_matrix_normalized = normalize(test_matrix, norm='l2', axis=1)
        
        test_minhashes = []
        for vec in test_matrix_normalized:
            mh = MinHash(num_perm=128)
            for idx in vec.nonzero()[1]:
                mh.update(str(idx).encode('utf-8'))
            test_minhashes.append(mh)
        
        test_predictions = {}
        for idx, mh in enumerate(test_minhashes):
            results = lsh.query(mh)
            similar = [train_data['id'][r] for r in results]
            test_predictions[test_data['id'][idx]] = similar[:5]
        
        return test_predictions
    
    # Example usage for test data
    # test_predictions = predict_test('test_ids.txt', 'test_texts.txt')
    # Save predictions to a file
    # with open('test_predictions.json', 'w') as f:
    #     json.dump(test_predictions, f)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from datasketch import MinHash, MinHashLSH
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# 1. Improved Data Loading and Preprocessing
def load_data(ids_file, texts_file, items_file=None):
    with open(ids_file, 'r') as f:
        ids = [line.strip() for line in f]
    with open(texts_file, 'r') as f:
        texts = [line.strip() for line in f]
    
    data = pd.DataFrame({'id': ids, 'text': texts})
    
    if items_file:
        with open(items_file, 'r') as f:
            ground_truth = json.load(f)
        return data, ground_truth
    return data

# 2. Enhanced Feature Engineering
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def engineer_features(texts):
    # Preprocess texts
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # TF-IDF Vectorization with improved parameters
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
        max_features=20000,  # Increase number of features
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of the documents
        sublinear_tf=True  # Apply sublinear tf scaling
    )
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
    
    # Normalize the TF-IDF matrix
    normalized_matrix = normalize(tfidf_matrix, norm='l2', axis=1)
    
    return normalized_matrix, vectorizer

# 3. Improved LSH Implementation
def create_minhash_lsh(matrix, num_perm=256):
    minhashes = []
    for vec in matrix:
        mh = MinHash(num_perm=num_perm)
        for idx in vec.nonzero()[1]:
            mh.update(str(idx).encode('utf-8'))
        minhashes.append(mh)
    
    lsh = MinHashLSH(threshold=0.7, num_perm=num_perm)
    for idx, mh in enumerate(minhashes):
        lsh.insert(idx, mh)
    
    return lsh, minhashes

# 4. Improved Similarity Computation
def find_similar_items(lsh, minhashes, k=5):
    similar_items = {}
    for idx, mh in enumerate(minhashes):
        results = lsh.query(mh)
        similar = [r for r in results if r != idx]
        if len(similar) < k:
            # If not enough results, use brute force to find more
            distances = [(i, minhashes[idx].jaccard(minhashes[i])) for i in range(len(minhashes)) if i != idx]
            distances.sort(key=lambda x: x[1], reverse=True)
            similar.extend([i for i, _ in distances if i not in similar])
        similar_items[idx] = similar[:k]
    return similar_items

# 5. Evaluation
def evaluate(predictions, ground_truth):
    intersection_scores = []
    for idx, pred in predictions.items():
        gt = ground_truth.get(str(idx), [])
        intersection = set(pred).intersection(set(gt))
        intersection_scores.append(len(intersection))
    return intersection_scores

# 6. Visualization
def visualize_results(intersection_scores):
    plt.figure(figsize=(12, 4))
    
    # Histogram
    plt.subplot(131)
    plt.hist(intersection_scores, bins=6, range=(0, 5))
    plt.title('Histogram of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.ylabel('Frequency')
    
    # Box Plot
    plt.subplot(132)
    plt.boxplot(intersection_scores)
    plt.title('Box Plot of Intersection Scores')
    plt.ylabel('Intersection Score')
    
    # Statistics
    plt.subplot(133)
    stats = pd.Series(intersection_scores).describe()
    plt.axis('off')
    plt.text(0.1, 0.9, str(stats), fontsize=10, verticalalignment='top')
    plt.title('Statistics of Intersection Scores')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    train_data, ground_truth = load_data('ids.txt', 'texts.txt', 'items.json')
    
    # Feature engineering
    matrix, vectorizer = engineer_features(train_data['text'])
    
    # Create LSH index
    lsh, minhashes = create_minhash_lsh(matrix)
    
    # Find similar items
    similar_items = find_similar_items(lsh, minhashes)
    
    # Evaluate
    intersection_scores = evaluate(similar_items, ground_truth)
    
    # Visualize results
    visualize_results(intersection_scores)
    
    # Calculate average score
    average_score = np.mean(intersection_scores)
    print(f"Average Intersection Score: {average_score:.2f}")
    
    # Save the model
    save_model(vectorizer, lsh, minhashes)

    # Function to predict for test data
    def predict_test(test_ids_file, test_texts_file):
        test_data = load_data(test_ids_file, test_texts_file)
        test_matrix = vectorizer.transform([preprocess_text(text) for text in test_data['text']])
        test_matrix_normalized = normalize(test_matrix, norm='l2', axis=1)
        
        test_minhashes = []
        for vec in test_matrix_normalized:
            mh = MinHash(num_perm=256)
            for idx in vec.nonzero()[1]:
                mh.update(str(idx).encode('utf-8'))
            test_minhashes.append(mh)
        
        test_predictions = {}
        for idx, mh in enumerate(test_minhashes):
            results = lsh.query(mh)
            similar = [train_data['id'][r] for r in results]
            if len(similar) < 5:
                distances = [(i, mh.jaccard(minhashes[i])) for i in range(len(minhashes))]
                distances.sort(key=lambda x: x[1], reverse=True)
                similar.extend([train_data['id'][i] for i, _ in distances if train_data['id'][i] not in similar])
            test_predictions[test_data['id'][idx]] = similar[:5]
        
        return test_predictions
    
    # Example usage for test data
    # test_predictions = predict_test('test_ids.txt', 'test_texts.txt')
    # Save predictions to a file
    # with open('test_predictions.json', 'w') as f:
    #     json.dump(test_predictions, f)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from datasketch import MinHash, MinHashLSH
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# 1. Improved Data Loading and Preprocessing
def load_data(ids_file, texts_file, items_file=None):
    with open(ids_file, 'r') as f:
        ids = [line.strip() for line in f]
    with open(texts_file, 'r') as f:
        texts = [line.strip() for line in f]
    
    data = pd.DataFrame({'id': ids, 'text': texts})
    
    if items_file:
        with open(items_file, 'r') as f:
            ground_truth = json.load(f)
        return data, ground_truth
    return data

# 2. Enhanced Feature Engineering
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def engineer_features(texts):
    # Preprocess texts
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # TF-IDF Vectorization with improved parameters
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
        max_features=20000,  # Increase number of features
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of the documents
        sublinear_tf=True  # Apply sublinear tf scaling
    )
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
    
    # Normalize the TF-IDF matrix
    normalized_matrix = normalize(tfidf_matrix, norm='l2', axis=1)
    
    return normalized_matrix, vectorizer

# 3. Improved LSH Implementation
def create_minhash_lsh(matrix, num_perm=256):
    minhashes = []
    for vec in matrix:
        mh = MinHash(num_perm=num_perm)
        for idx in vec.nonzero()[1]:
            mh.update(str(idx).encode('utf-8'))
        minhashes.append(mh)
    
    lsh = MinHashLSH(threshold=0.7, num_perm=num_perm)
    for idx, mh in enumerate(minhashes):
        lsh.insert(idx, mh)
    
    return lsh, minhashes

# 4. Improved Similarity Computation
def find_similar_items(lsh, minhashes, k=5):
    similar_items = {}
    for idx, mh in enumerate(minhashes):
        results = lsh.query(mh)
        similar = [r for r in results if r != idx]
        if len(similar) < k:
            # If not enough results, use brute force to find more
            distances = [(i, minhashes[idx].jaccard(minhashes[i])) for i in range(len(minhashes)) if i != idx]
            distances.sort(key=lambda x: x[1], reverse=True)
            similar.extend([i for i, _ in distances if i not in similar])
        similar_items[idx] = similar[:k]
    return similar_items

# 5. Evaluation
def evaluate(predictions, ground_truth):
    intersection_scores = []
    for idx, pred in predictions.items():
        gt = ground_truth.get(str(idx), [])
        intersection = set(pred).intersection(set(gt))
        intersection_scores.append(len(intersection))
    return intersection_scores

# 6. Visualization
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from datasketch import MinHash, MinHashLSH
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ... (previous code remains the same) ...

# Update the visualize_results function to save the graph
def visualize_results(intersection_scores, save_path='intersection_scores_plot.png'):
    plt.figure(figsize=(12, 4))
    
    # Histogram
    plt.subplot(131)
    plt.hist(intersection_scores, bins=6, range=(0, 5))
    plt.title('Histogram of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.ylabel('Frequency')
    
    # Box Plot
    plt.subplot(132)
    plt.boxplot(intersection_scores)
    plt.title('Box Plot of Intersection Scores')
    plt.ylabel('Intersection Score')
    
    # Statistics
    plt.subplot(133)
    stats = pd.Series(intersection_scores).describe()
    plt.axis('off')
    plt.text(0.1, 0.9, str(stats), fontsize=10, verticalalignment='top')
    plt.title('Statistics of Intersection Scores')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")

# Main execution
if __name__ == "__main__":
    # Load data
    train_data, ground_truth = load_data('ids.txt', 'texts.txt', 'items.json')
    
    # Feature engineering
    matrix, vectorizer = engineer_features(train_data['text'])
    
    # Create LSH index
    lsh, minhashes = create_minhash_lsh(matrix)
    
    # Find similar items
    similar_items = find_similar_items(lsh, minhashes)
    
    # Evaluate
    intersection_scores = evaluate(similar_items, ground_truth)
    
    # Visualize results and save the graph
    visualize_results(intersection_scores)
    
    # Calculate average score
    average_score = np.mean(intersection_scores)
    print(f"Average Intersection Score: {average_score:.2f}")
    
    # Save the model
    save_model(vectorizer, lsh, minhashes)

    # Save intersection scores to a CSV file
    scores_df = pd.DataFrame({'intersection_score': intersection_scores})
    scores_df.to_csv('intersection_scores.csv', index=False)
    print("Intersection scores saved to intersection_scores.csv")

    # Save similar items to a JSON file
    with open('similar_items.json', 'w') as f:
        json.dump({str(k): [str(i) for i in v] for k, v in similar_items.items()}, f)
    print("Similar items saved to similar_items.json")

    # Function to predict for test data
    def predict_test(test_ids_file, test_texts_file):
        test_data = load_data(test_ids_file, test_texts_file)
        test_matrix = vectorizer.transform([preprocess_text(text) for text in test_data['text']])
        test_matrix_normalized = normalize(test_matrix, norm='l2', axis=1)
        
        test_minhashes = []
        for vec in test_matrix_normalized:
            mh = MinHash(num_perm=256)
            for idx in vec.nonzero()[1]:
                mh.update(str(idx).encode('utf-8'))
            test_minhashes.append(mh)
        
        test_predictions = {}
        for idx, mh in enumerate(test_minhashes):
            results = lsh.query(mh)
            similar = [train_data['id'][r] for r in results]
            if len(similar) < 5:
                distances = [(i, mh.jaccard(minhashes[i])) for i in range(len(minhashes))]
                distances.sort(key=lambda x: x[1], reverse=True)
                similar.extend([train_data['id'][i] for i, _ in distances if train_data['id'][i] not in similar])
            test_predictions[test_data['id'][idx]] = similar[:5]
        
        return test_predictions
    
    # Example usage for test data
    # test_predictions = predict_test('test_ids.txt', 'test_texts.txt')
    # Save predictions to a file
    # with open('test_predictions.json', 'w') as f:
    #     json.dump(test_predictions, f)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from datasketch import MinHash, MinHashLSH
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# 1. Data Loading and Preprocessing
def load_data(ids_file, texts_file, items_file=None):
    with open(ids_file, 'r') as f:
        ids = [line.strip() for line in f]
    with open(texts_file, 'r') as f:
        texts = [line.strip() for line in f]
    
    data = pd.DataFrame({'id': ids, 'text': texts})
    
    if items_file:
        with open(items_file, 'r') as f:
            ground_truth = json.load(f)
        return data, ground_truth
    return data

# 2. Feature Engineering
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def engineer_features(texts):
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=20000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
    
    normalized_matrix = normalize(tfidf_matrix, norm='l2', axis=1)
    
    return normalized_matrix, vectorizer

# 3. LSH Implementation
def create_minhash_lsh(matrix, num_perm=256):
    minhashes = []
    for vec in matrix:
        mh = MinHash(num_perm=num_perm)
        for idx in vec.nonzero()[1]:
            mh.update(str(idx).encode('utf-8'))
        minhashes.append(mh)
    
    lsh = MinHashLSH(threshold=0.7, num_perm=num_perm)
    for idx, mh in enumerate(minhashes):
        lsh.insert(idx, mh)
    
    return lsh, minhashes

# 4. Similarity Computation
# def find_similar_items(lsh, minhashes, k=5):
#     similar_items = {}
#     for idx, mh in enumerate(minhashes):
#         results = lsh.query(mh)
#         similar = [r for r in results if r != idx]
#         if len(similar) < k:
#             distances = [(i, minhashes[idx].jaccard(minhashes[i])) for i in range(len(minhashes)) if i != idx]
#             distances.sort(key=lambda x: x[1], reverse=True)
#             similar.extend([i for i, _ in distances if i not in similar])
#         similar_items[idx] = similar[:k]
#     return similar_items

def find_similar_items(lsh, minhashes, matrix, k=5):
    print("Finding similar items...")
    similar_items = {}
    matrix_csr = csr_matrix(matrix)
    
    for idx, mh in enumerate(minhashes):
        if idx % 1000 == 0:
            print(f"Processing item {idx}")
        
        results = lsh.query(mh)
        similar = [r for r in results if r != idx]
        
        if len(similar) < k:
            # Use cosine similarity on TF-IDF vectors for remaining items
            remaining = set(range(len(minhashes))) - set(similar) - {idx}
            if remaining:
                cosine_sims = cosine_similarity(matrix_csr[idx], matrix_csr[list(remaining)])
                top_indices = cosine_sims.argsort()[0][-k:][::-1]
                similar.extend([list(remaining)[i] for i in top_indices])
        
        similar_items[idx] = similar[:k]
    
    print("Similar items found successfully.")
    return similar_items


# 5. Evaluation
def evaluate(predictions, ground_truth):
    intersection_scores = []
    for idx, pred in predictions.items():
        gt = ground_truth.get(str(idx), [])
        intersection = set(pred).intersection(set(gt))
        intersection_scores.append(len(intersection))
    return intersection_scores

# 6. Visualization
def visualize_results(intersection_scores, save_path='intersection_scores_plot.png'):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.hist(intersection_scores, bins=6, range=(0, 5))
    plt.title('Histogram of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.ylabel('Frequency')
    
    plt.subplot(132)
    plt.boxplot(intersection_scores)
    plt.title('Box Plot of Intersection Scores')
    plt.ylabel('Intersection Score')
    
    plt.subplot(133)
    stats = pd.Series(intersection_scores).describe()
    plt.axis('off')
    plt.text(0.1, 0.9, str(stats), fontsize=10, verticalalignment='top')
    plt.title('Statistics of Intersection Scores')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")

# Function to save the model
def save_model(vectorizer, lsh, minhashes, filename='similarity_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((vectorizer, lsh, minhashes), f)
    print(f"Model saved to {filename}")

# Function to load the model
def load_model(filename='similarity_model.pkl'):
    with open(filename, 'rb') as f:
        vectorizer, lsh, minhashes = pickle.load(f)
    return vectorizer, lsh, minhashes

# ... (previous code remains the same) ...

# Main execution
# ... (imports and other functions remain the same) ...

# Main execution
if __name__ == "__main__":
    print("Starting similarity search process...")

    # Load data
    print("Loading data...")
    train_data, ground_truth = load_data('ids.txt', 'texts.txt', 'items.json')
    print("Data loaded successfully.")
    
    # Feature engineering
    print("Performing feature engineering...")
    matrix, vectorizer = engineer_features(train_data['text'])
    print("Feature engineering completed.")
    
    # Create LSH index
    print("Creating LSH index...")
    lsh, minhashes = create_minhash_lsh(matrix)
    print("LSH index created successfully.")
    print(len(minhashes))
    # Find similar items
    print("Finding similar items...")
    similar_items = find_similar_items(lsh, minhashes, matrix)  # Pass matrix here
    print("Similar items found successfully.")
    
    # Evaluate
    print("Evaluating results...")
    intersection_scores = evaluate(similar_items, ground_truth)
    print("Evaluation completed.")
    
    # Visualize results and save the graph
    print("Generating visualization...")
    visualize_results(intersection_scores)
    print("Visualization saved successfully.")
    
    # Calculate average score
    average_score = np.mean(intersection_scores)
    print(f"Average Intersection Score: {average_score:.2f}")
    
    # Save the model
    print("Saving the model...")
    save_model(vectorizer, lsh, minhashes)
    print("Model saved successfully.")

    # Save intersection scores to a CSV file
    print("Saving intersection scores...")
    scores_df = pd.DataFrame({'intersection_score': intersection_scores})
    scores_df.to_csv('intersection_scores.csv', index=False)
    print("Intersection scores saved to intersection_scores.csv")

    # Save similar items to a JSON file
    print("Saving similar items...")
    with open('similar_items.json', 'w') as f:
        json.dump({str(k): [str(i) for i in v] for k, v in similar_items.items()}, f)
    print("Similar items saved to similar_items.json")

    print("All processes completed successfully.")

    # Function to predict for test data
    def predict_test(test_ids_file, test_texts_file):
        print("Starting prediction for test data...")
        test_data = load_data(test_ids_file, test_texts_file)
        print("Test data loaded.")
        
        print("Transforming test data...")
        test_matrix = vectorizer.transform([preprocess_text(text) for text in test_data['text']])
        test_matrix_normalized = normalize(test_matrix, norm='l2', axis=1)
        print("Test data transformed.")
        
        print("Creating MinHash signatures for test data...")
        test_minhashes = []
        for vec in test_matrix_normalized:
            mh = MinHash(num_perm=256)
            for idx in vec.nonzero()[1]:
                mh.update(str(idx).encode('utf-8'))
            test_minhashes.append(mh)
        print("MinHash signatures created.")
        
        print("Finding similar items for test data...")
        test_predictions = find_similar_items(lsh, test_minhashes, test_matrix_normalized)
        print("Similar items found for test data.")
        
        # Convert index-based predictions to id-based predictions
        id_based_predictions = {
            test_data['id'][idx]: [train_data['id'][i] for i in similar]
            for idx, similar in test_predictions.items()
        }
        
        return id_based_predictions
    
    # Example usage for test data
    # print("Starting prediction for test data...")
    # test_predictions = predict_test('test_ids.txt', 'test_texts.txt')
    # print("Test predictions completed.")
    # Save predictions to a file
    # print("Saving test predictions...")
    # with open('test_predictions.json', 'w') as f:
    #     json.dump(test_predictions, f)
    # print("Test predictions saved to test_predictions.json")
import pandas as pd
import json

# Load ids and texts
with open('ids.txt', 'r') as f:
    ids = f.read().splitlines()

with open('texts.txt', 'r') as f:
    texts = f.read().splitlines()

# Print lengths for debugging
print(f"Number of ids: {len(ids)}")
print(f"Number of texts: {len(texts)}")

# Ensure ids and texts have the same length
min_length = min(len(ids), len(texts))
ids = ids[:min_length]
texts = texts[:min_length]

# Create a DataFrame
df = pd.DataFrame({'id': ids, 'text': texts})

# Load items.json
with open('items.json', 'r') as f:
    items = json.load(f)

# Print DataFrame info for verification
print(df.info())



import pandas as pd
import numpy as np
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading and Preprocessing
def load_data(ids_file, texts_file, items_file):
    with open(ids_file, 'r') as f:
        ids = f.read().splitlines()
    with open(texts_file, 'r') as f:
        texts = f.read().splitlines()
    
    min_length = min(len(ids), len(texts))
    ids = ids[:min_length]
    texts = texts[:min_length]
    
    df = pd.DataFrame({'id': ids, 'text': texts})
    
    with open(items_file, 'r') as f:
        items = json.load(f)
    
    return df, items

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Feature Engineering
def engineer_features(df):
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
    
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df['text'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))
    df['unique_word_count'] = df['text'].apply(lambda x: len(set(x.split())))
    df['unique_word_ratio'] = df['unique_word_count'] / df['word_count']
    
    return df, tfidf_matrix, tfidf_vectorizer

# LSH Implementation
def create_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

def build_lsh_index(df, num_perm=128, threshold=0.5):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for idx, row in df.iterrows():
        minhash = create_minhash(row['processed_text'], num_perm)
        lsh.insert(row['id'], minhash)
    return lsh

def find_similar_items(lsh, df, query_id, num_results=5):
    query_minhash = create_minhash(df.loc[df['id'] == query_id, 'processed_text'].iloc[0])
    return lsh.query(query_minhash)[:num_results]

# Evaluation
def evaluate_model(df, lsh, ground_truth):
    intersection_scores = []
    for idx, row in df.iterrows():
        predicted = find_similar_items(lsh, df, row['id'])
        actual = ground_truth.get(row['id'], [])
        intersection = set(predicted) & set(actual)
        intersection_scores.append(len(intersection))
    return intersection_scores

# Visualization
def visualize_results(intersection_scores):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.hist(intersection_scores, bins=6, edgecolor='black')
    plt.title('Histogram of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.ylabel('Frequency')
    
    plt.subplot(122)
    sns.boxplot(intersection_scores)
    plt.title('Box Plot of Intersection Scores')
    
    plt.tight_layout()
    plt.show()
    
    print(pd.Series(intersection_scores).describe())

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df, ground_truth = load_data('ids.txt', 'texts.txt', 'items.json')
    df, tfidf_matrix, tfidf_vectorizer = engineer_features(df)
    
    # Build LSH index
    lsh = build_lsh_index(df)
    
    # Evaluate model
    intersection_scores = evaluate_model(df, lsh, ground_truth)
    
    # Visualize results
    visualize_results(intersection_scores)
    
    # Calculate average score
    average_score = sum(intersection_scores) / len(intersection_scores)
    print(f"Average Intersection Score: {average_score:.2f}")

    # Function to handle test data
    def process_test_data(test_ids_file, test_texts_file):
        test_df = load_data(test_ids_file, test_texts_file, None)[0]
        test_df['processed_text'] = test_df['text'].apply(preprocess_text)
        test_tfidf = tfidf_vectorizer.transform(test_df['processed_text'])
        return test_df

    # Example usage for test data
    # test_df = process_test_data('test_ids.txt', 'test_texts.txt')
    # Now you can use this test_df with your LSH model to find similar items
import pandas as pd
import numpy as np
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from datasketch import MinHash, MinHashLSH
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Data Loading and Preprocessing
def load_data(ids_file, texts_file, items_file):
    with open(ids_file, 'r') as f:
        ids = f.read().splitlines()
    with open(texts_file, 'r') as f:
        texts = f.read().splitlines()
    
    min_length = min(len(ids), len(texts))
    ids = ids[:min_length]
    texts = texts[:min_length]
    
    df = pd.DataFrame({'id': ids, 'text': texts})
    
    with open(items_file, 'r') as f:
        items = json.load(f)
    
    return df, items

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Improved Feature Engineering
# Improved Feature Engineering
def engineer_features(df):
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
    
    # LSA to reduce dimensionality
    lsa = TruncatedSVD(n_components=300, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)
    
    # Basic text features
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df['text'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))
    df['unique_word_count'] = df['text'].apply(lambda x: len(set(x.split())))
    df['unique_word_ratio'] = df['unique_word_count'] / df['word_count']
    
    # Advanced text features
    df['sentence_count'] = df['text'].apply(lambda x: len(re.findall(r'\w+[.!?]', x)))
    df['avg_sentence_length'] = df['word_count'] / df['sentence_count'].replace(0, 1)  # Avoid division by zero
    
    # Part-of-speech tagging
    from nltk import pos_tag
    
    def safe_pos_count(text, pos_prefix):
        try:
            return len([word for word, pos in pos_tag(word_tokenize(text)) if pos.startswith(pos_prefix)])
        except Exception as e:
            print(f"Error in POS tagging: {e}")
            return 0
    
    df['noun_count'] = df['processed_text'].apply(lambda x: safe_pos_count(x, 'N'))
    df['verb_count'] = df['processed_text'].apply(lambda x: safe_pos_count(x, 'V'))
    df['adj_count'] = df['processed_text'].apply(lambda x: safe_pos_count(x, 'J'))
    
    # Named Entity Recognition
    from nltk import ne_chunk
    
    def safe_ne_count(text):
        try:
            return len([chunk for chunk in ne_chunk(pos_tag(word_tokenize(text))) if hasattr(chunk, 'label')])
        except Exception as e:
            print(f"Error in NER: {e}")
            return 0
    
    df['named_entity_count'] = df['processed_text'].apply(safe_ne_count)
    
    # Combine all features
    feature_matrix = np.hstack((lsa_matrix, df[['text_length', 'word_count', 'avg_word_length', 'unique_word_count', 'unique_word_ratio', 
                                               'sentence_count', 'avg_sentence_length', 'noun_count', 'verb_count', 'adj_count', 'named_entity_count']].values))
    
    return df, feature_matrix, tfidf_vectorizer

# Improved LSH Implementation
def create_minhash(features, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for i, feature in enumerate(features):
        m.update(f"{i}:{feature}".encode('utf8'))
    return m

def build_lsh_index(df, feature_matrix, num_perm=128, threshold=0.5):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, row in df.iterrows():
        minhash = create_minhash(feature_matrix[i], num_perm)
        lsh.insert(row['id'], minhash)
    return lsh

def find_similar_items(lsh, df, feature_matrix, query_id, num_results=5):
    query_index = df.index[df['id'] == query_id][0]
    query_minhash = create_minhash(feature_matrix[query_index])
    return lsh.query(query_minhash)[:num_results]

# Evaluation
def evaluate_model(df, lsh, feature_matrix, ground_truth):
    intersection_scores = []
    for idx, row in df.iterrows():
        predicted = find_similar_items(lsh, df, feature_matrix, row['id'])
        actual = ground_truth.get(row['id'], [])
        intersection = set(predicted) & set(actual)
        intersection_scores.append(len(intersection))
    return intersection_scores

# Visualization
def visualize_results(intersection_scores):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.hist(intersection_scores, bins=6, edgecolor='black')
    plt.title('Histogram of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.ylabel('Frequency')
    
    plt.subplot(122)
    sns.boxplot(intersection_scores)
    plt.title('Box Plot of Intersection Scores')
    
    plt.tight_layout()
    plt.show()
    
    print(pd.Series(intersection_scores).describe())

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df, ground_truth = load_data('ids.txt', 'texts.txt', 'items.json')
    df, feature_matrix, tfidf_vectorizer = engineer_features(df)
    
    # Build LSH index
    lsh = build_lsh_index(df, feature_matrix)
    
    # Evaluate model
    intersection_scores = evaluate_model(df, lsh, feature_matrix, ground_truth)
    
    # Visualize results
    visualize_results(intersection_scores)
    
    # Calculate average score
    average_score = sum(intersection_scores) / len(intersection_scores)
    print(f"Average Intersection Score: {average_score:.2f}")

    # Function to handle test data
    def process_test_data(test_ids_file, test_texts_file):
        test_df, _ = load_data(test_ids_file, test_texts_file, None)
        test_df, test_feature_matrix, _ = engineer_features(test_df)
        return test_df, test_feature_matrix

    # Example usage for test data
    # test_df, test_feature_matrix = process_test_data('test_ids.txt', 'test_texts.txt')
    # Now you can use this test_df and test_feature_matrix with your LSH model to find similar items
import pandas as pd
import json
from datasketch import MinHash, MinHashLSH
import matplotlib.pyplot as plt
from typing import Set, Dict, List
import numpy as np

# Constants
SIMILARITY_THRESHOLD = 0.6
NUM_PERMS = 96
SHINGLE_SIZE = 4

def load_data() -> pd.DataFrame:
    """Load and combine ID and text data."""
    try:
        ids = pd.read_csv('ids.txt', header=None, names=['id'])
        texts = pd.read_csv('texts.txt', header=None, names=['text'])
        return pd.concat([ids, texts], axis=1)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find input files: {str(e)}")
    except pd.errors.EmptyDataError:
        raise ValueError("One or both input files are empty")

def create_shingles(text: str, size: int) -> Set[str]:
    """Create shingles from input text."""
    if not isinstance(text, str):
        return set()
    return {text[i:i+size] for i in range(len(text) - size + 1)}

def create_minhash_index(data: pd.DataFrame) -> tuple[MinHashLSH, Dict]:
    """Create MinHash LSH index and minhashes dictionary."""
    lsh = MinHashLSH(threshold=SIMILARITY_THRESHOLD, num_perm=NUM_PERMS)
    minhashes = {}
    
    for idx, row in data.iterrows():
        if pd.isna(row['text']):
            continue
            
        shingles = create_shingles(str(row['text']), SHINGLE_SIZE)
        if not shingles:
            continue
            
        minhash = MinHash(num_perm=NUM_PERMS)
        for shingle in shingles:
            minhash.update(shingle.encode('utf8'))
            
        lsh.insert(str(row['id']), minhash)  # Convert id to string for consistency
        minhashes[str(row['id'])] = minhash
        
    return lsh, minhashes

def find_similar_items(data: pd.DataFrame, lsh: MinHashLSH, 
                      minhashes: Dict, top_k: int = 5) -> Dict:
    """Find similar items for each document."""
    results = {}
    for idx, row in data.iterrows():
        doc_id = str(row['id'])
        if doc_id not in minhashes:
            continue
            
        similar_items = lsh.query(minhashes[doc_id])
        # Remove self from similar items
        similar_items = [item for item in similar_items if item != doc_id]
        results[doc_id] = similar_items[:top_k]
    
    return results

def calculate_intersection_scores(results: Dict, ground_truth_file: str) -> List[int]:
    """Calculate intersection scores with ground truth."""
    try:
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Ground truth file {ground_truth_file} not found")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in ground truth file")

    intersection_scores = []
    for id_, similar_ids in results.items():
        true_similar_ids = set(str(id_) for id_ in ground_truth.get(id_, []))
        intersection_count = len(set(similar_ids) & true_similar_ids)
        intersection_scores.append(intersection_count)
    
    return intersection_scores

def plot_results(intersection_scores: List[int]) -> None:
    """Create visualization of results."""
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(121)
    plt.hist(intersection_scores, bins=range(max(intersection_scores) + 2), 
             alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Histogram of Intersection Scores')
    plt.xlabel('Intersection Score')
    plt.ylabel('Frequency')
    
    # Box Plot
    plt.subplot(122)
    plt.boxplot(intersection_scores)
    plt.title('Box Plot of Intersection Scores')
    plt.ylabel('Intersection Score')
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Load and process data
        data = load_data()
        lsh, minhashes = create_minhash_index(data)
        results = find_similar_items(data, lsh, minhashes)
        
        # Calculate and visualize results
        intersection_scores = calculate_intersection_scores(results, 'items.json')
        plot_results(intersection_scores)
        
        # Print statistics
        stats = pd.Series(intersection_scores).describe()
        print("\nIntersection Score Statistics:")
        print(stats)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
import pandas as pd
import json
import numpy as np
from datasketch import MinHash, MinHashLSH
import matplotlib.pyplot as plt

# Load IDs and texts
ids = pd.read_csv('ids.txt', header=None, names=['id'])
texts = pd.read_csv('texts.txt', header=None, names=['text'], sep='\n', engine='python')

with open('texts.txt', 'r') as file:
    paragraphs = file.read().split('\n\n')  # Assumes paragraphs are separated by blank lines

texts = pd.DataFrame(paragraphs, columns=['text'])


# Load ground truth data
with open('items.json') as f:
    ground_truth = json.load(f)

import re

def parse_data(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    
    parsed_data = []
    for line in data:
        id_, text = line.split(',', 1)  # assuming each line is "id,text"
        processed_text = re.sub(r'\W+', '', text.lower())  # remove punctuation and lowercase
        parsed_data.append((id_, processed_text))
    
    return parsed_data

import binascii

def shingle_document(text, k):
    shingles = set()
    for i in range(len(text) - k + 1):
        shingle = text[i:i+k]
        hashed_shingle = binascii.crc32(shingle.encode('utf-8')) & 0xffffffff  # 32-bit int
        shingles.add(hashed_shingle)
    return shingles

def jaccard(shingles1, shingles2):
    intersection = len(shingles1.intersection(shingles2))
    union = len(shingles1.union(shingles2))
    return intersection / union if union != 0 else 0


def calculate_similarities(data, k):
    shingles = {id_: shingle_document(text, k) for id_, text in data}
    similarities = []
    
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            id1, id2 = data[i][0], data[j][0]
            jaccard_sim = jaccard(shingles[id1], shingles[id2])
            similarities.append((id1, id2, jaccard_sim))
    
    return similarities

import matplotlib.pyplot as plt

# Use different values of k, say [3, 5, 7, 9]
ks = [3, 5, 7, 9]
plagiarism_sims = []
non_plagiarism_sims = []

for k in ks:
    sims = calculate_similarities(parsed_data, k)
    # assume you know which ids are plagiarism instances
    plagiarism_sims.append(np.mean([s for id1, id2, s in sims if is_plagiarism(id1, id2)]))
    non_plagiarism_sims.append(np.mean([s for id1, id2, s in sims if not is_plagiarism(id1, id2)]))

plt.plot(ks, plagiarism_sims, label="Plagiarism")
plt.plot(ks, non_plagiarism_sims, label="Non-plagiarism")
plt.xlabel("Sharding length (k)")
plt.ylabel("Jaccard similarity")
plt.legend()
plt.show()

def invert_shingles(shingled_documents):
    inv_index = []
    docids = []
    
    for docid, shingles in shingled_documents:
        docids.append(docid)
        for shingle in shingles:
            inv_index.append((shingle, docid))
    
    inv_index.sort()  # sort by shingle (item)
    return inv_index, docids

import numpy as np

def make_minhash_signature(shingled_data, num_hashes):
    inv_index, docids = invert_shingles(shingled_data)
    num_docs = len(docids)
    sigmatrix = np.full((num_hashes, num_docs), np.inf)
    
    hash_funcs = [make_random_hash_fn() for _ in range(num_hashes)]
    
    for shingle, docid in inv_index:
        for i, hash_fn in enumerate(hash_funcs):
            hash_val = hash_fn(shingle)
            doc_idx = docids.index(docid)
            sigmatrix[i, doc_idx] = min(sigmatrix[i, doc_idx], hash_val)
    
    return sigmatrix, docids

def minhash_similarity(id1, id2, minhash_sigmat, docids):
    idx1 = docids.index(id1)
    idx2 = docids.index(id2)
    
    sig1 = minhash_sigmat[:, idx1]
    sig2 = minhash_sigmat[:, idx2]
    
    return np.mean(sig1 == sig2)

from collections import defaultdict

def do_lsh(minhash_sigmatrix, numhashes, docids, threshold):
    b, _ = choose_nbands(threshold, numhashes)
    r = int(numhashes / b)
    
    buckets = []
    for band in range(b):
        cur_buckets = defaultdict(list)
        for i in range(len(docids)):
            band_vector = tuple(minhash_sigmatrix[band*r:(band+1)*r, i])
            cur_buckets[band_vector].append(docids[i])
        buckets.append(cur_buckets)
    
    return buckets

def get_lsh_candidates(buckets):
    candidates = set()
    for bucket in buckets:
        for doc_ids in bucket.values():
            if len(doc_ids) > 1:
                for i in range(len(doc_ids)):
                    for j in range(i + 1, len(doc_ids)):
                        candidates.add((doc_ids[i], doc_ids[j]))
    return list(candidates)


import re
import binascii
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ------------------- Step 1: Data Parsing -------------------
def parse_data(ids_file, texts_file):
    with open(ids_file, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    
    with open(texts_file, 'r') as f:
        texts = [re.sub(r'\W+', '', line.strip().lower()) for line in f.readlines()]
    
    return list(zip(ids, texts))

# ------------------- Step 2: Shingling -------------------
def shingle_document(text, k):
    shingles = set()
    for i in range(len(text) - k + 1):
        shingle = text[i:i+k]
        hashed_shingle = binascii.crc32(shingle.encode('utf-8')) & 0xffffffff  # 32-bit int
        shingles.add(hashed_shingle)
    return shingles

def jaccard(shingles1, shingles2):
    intersection = len(shingles1.intersection(shingles2))
    union = len(shingles1.union(shingles2))
    return intersection / union if union != 0 else 0

# ------------------- Step 3: MinHashing -------------------
def make_random_hash_fn():
    a, b = np.random.randint(1, 1e6, size=2)
    return lambda x: (a * x + b) % 1000003  # A large prime number

def invert_shingles(shingled_documents):
    inv_index = []
    docids = []
    
    for docid, shingles in shingled_documents:
        docids.append(docid)
        for shingle in shingles:
            inv_index.append((shingle, docid))
    
    inv_index.sort()  # sort by shingle (item)
    return inv_index, docids

def make_minhash_signature(shingled_data, num_hashes):
    inv_index, docids = invert_shingles(shingled_data)
    num_docs = len(docids)
    sigmatrix = np.full((num_hashes, num_docs), np.inf)
    
    hash_funcs = [make_random_hash_fn() for _ in range(num_hashes)]
    
    for shingle, docid in inv_index:
        for i, hash_fn in enumerate(hash_funcs):
            hash_val = hash_fn(shingle)
            doc_idx = docids.index(docid)
            sigmatrix[i, doc_idx] = min(sigmatrix[i, doc_idx], hash_val)
    
    return sigmatrix, docids

# ------------------- Step 4: Locality-Sensitive Hashing (LSH) -------------------
def choose_nbands(threshold, numhashes):
    b = int(np.log(1 / threshold))  # Number of bands
    r = int(numhashes / b)  # Rows per band
    return b, r

def do_lsh(minhash_sigmatrix, numhashes, docids, threshold):
    b, r = choose_nbands(threshold, numhashes)
    
    buckets = []
    for band in range(b):
        cur_buckets = defaultdict(list)
        for i in range(len(docids)):
            band_vector = tuple(minhash_sigmatrix[band*r:(band+1)*r, i])
            cur_buckets[band_vector].append(docids[i])
        buckets.append(cur_buckets)
    
    return buckets

def get_lsh_candidates(buckets):
    candidates = set()
    for bucket in buckets:
        for doc_ids in bucket.values():
            if len(doc_ids) > 1:
                for i in range(len(doc_ids)):
                    for j in range(i + 1, len(doc_ids)):
                        candidates.add((doc_ids[i], doc_ids[j]))
    return list(candidates)

def minhash_similarity(id1, id2, minhash_sigmat, docids):
    idx1 = docids.index(id1)
    idx2 = docids.index(id2)
    
    sig1 = minhash_sigmat[:, idx1]
    sig2 = minhash_sigmat[:, idx2]
    
    return np.mean(sig1 == sig2)

# ------------------- Step 5: Evaluation and Plotting -------------------
def evaluate_model(predictions, ground_truth):
    intersection_scores = []
    
    for id_ in predictions:
        predicted_top5 = set(predictions[id_])
        ground_truth_top5 = set(ground_truth.get(id_, []))
        intersection_score = len(predicted_top5.intersection(ground_truth_top5))
        intersection_scores.append(intersection_score)
    
    return intersection_scores

def plot_statistics(scores):
    df = pd.DataFrame({'scores': scores})
    print(df.describe())
    
    sns.histplot(df['scores'], bins=6)
    plt.show()
    
    sns.boxplot(x=df['scores'])
    plt.show()

# ------------------- Step 6: Main Code Execution -------------------
if __name__ == "__main__":
    # Load data
    ids_file = 'ids.txt'
    texts_file = 'texts.txt'
    ground_truth_file = 'items.json'
    
    # Parse data
    parsed_data = parse_data(ids_file, texts_file)
    
    # Shingling
    k = 5  # Shingle length (can experiment with different values)
    shingles = [(id_, shingle_document(text, k)) for id_, text in parsed_data]
    
    # MinHashing
    num_hashes = 100  # Number of MinHash functions
    minhash_sigmatrix, docids = make_minhash_signature(shingles, num_hashes)
    
    # LSH
    threshold = 0.5  # Tune this parameter
    buckets = do_lsh(minhash_sigmatrix, num_hashes, docids, threshold)
    candidates = get_lsh_candidates(buckets)
    
    # Retrieve top 5 similar items for each document
    top5_predictions = defaultdict(list)
    
    for id1, id2 in candidates:
        similarity = minhash_similarity(id1, id2, minhash_sigmatrix, docids)
        top5_predictions[id1].append((id2, similarity))
        top5_predictions[id2].append((id1, similarity))
    
    # Sort predictions by similarity and select top 5
    for id_ in top5_predictions:
        top5_predictions[id_] = [item[0] for item in sorted(top5_predictions[id_], key=lambda x: x[1], reverse=True)[:5]]
    
    # Load ground truth
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    # Evaluate model
    intersection_scores = evaluate_model(top5_predictions, ground_truth)
    
    # Plot and print statistics
    plot_statistics(intersection_scores)

ids_file = 'ids.txt'
texts_file = 'texts.txt'
ground_truth_file = 'items.json'

# Parse data
parsed_data = parse_data(ids_file, texts_file)

# Shingling
k = 5  # Shingle length (can experiment with different values)
shingles = [(id_, shingle_document(text, k)) for id_, text in parsed_data]
    
print(len(shingles))
# MinHashing
num_hashes = 100  # Number of MinHash functions
minhash_sigmatrix, docids = make_minhash_signature(shingles, num_hashes)
import re
import binascii
import json
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Set, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ------------------- Part IA: Dataset Parsing -------------------
def parse_data(filename: str) -> List[Tuple[str, str]]:
    """Parse input file and process text according to requirements."""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                doc_id, text = parts
            else:
                doc_id, text = parts[0], ' '.join(parts[1:])
            # Remove punctuation, convert to lowercase, remove whitespace
            processed_text = re.sub(r'\W+', '', text.lower())
            data.append((doc_id, processed_text))
    return data

# ------------------- Part IB: Document Shingles -------------------
def shingle_document(text: str, k: int) -> Set[int]:
    """Create k-shingles from document and hash them."""
    shingles = set()
    for i in range(len(text) - k + 1):
        shingle = text[i:i+k]
        # Hash using CRC32 to get 32-bit integer
        hashed_shingle = binascii.crc32(shingle.encode('utf-8')) & 0xffffffff
        shingles.add(hashed_shingle)
    return shingles

# ------------------- Part IC: Jaccard Similarity -------------------
def jaccard(set1: Set[int], set2: Set[int]) -> float:
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# ------------------- Part ID: Computing All Similarities -------------------
def compute_all_similarities(data: List[Tuple[str, str]], k: int) -> List[Tuple[str, str, float]]:
    """Compute Jaccard similarities for all document pairs."""
    results = []
    shingled_docs = [(id_, shingle_document(text, k)) for id_, text in data]
    
    for i in range(len(shingled_docs)):
        for j in range(i + 1, len(shingled_docs)):
            id1, shingles1 = shingled_docs[i]
            id2, shingles2 = shingled_docs[j]
            sim = jaccard(shingles1, shingles2)
            results.append((id1, id2, sim))
    return results

# ------------------- Part IIA: Prepare Shingles for Processing -------------------
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

# ------------------- Part IIB: Generate Hash Functions -------------------
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
def choose_nbands(threshold: float, n: int) -> Tuple[int, int]:
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

# ------------------- Main Execution -------------------
def main():
    # Parameters
    k = 5  # shingle length
    num_hashes = 100  # number of hash functions
    threshold = 0.5  # LSH threshold
    
    # Parse input data
    data = parse_data('texts.txt')
    doc_ids = [id_ for id_, _ in data]
    
    # Create shingles
    shingled_data = [(id_, shingle_document(text, k)) for id_, text in data]
    
    # Generate MinHash signatures
    minhash_sigmatrix, docids = make_minhash_signature(shingled_data, num_hashes)
    
    # Perform LSH
    buckets = do_lsh(minhash_sigmatrix, num_hashes, docids, threshold)
    candidates = get_lsh_candidates(buckets)
    
    # Compute similarities for candidates
    results = []
    for id1, id2 in candidates:
        sim = minhash_similarity(id1, id2, minhash_sigmatrix, docids)
        if sim >= threshold:
            results.append((id1, id2, sim))
    
    # Load ground truth
    with open('items.json', 'r') as f:
        ground_truth = json.load(f)
    
    # Create predictions dictionary
    predictions = {}
    for id1 in doc_ids:
        similar_docs = []
        for id1_, id2_, sim in sorted(results, key=lambda x: x[2], reverse=True):
            if id1 == id1_:
                similar_docs.append(id2_)
            elif id1 == id2_:
                similar_docs.append(id1_)
            if len(similar_docs) == 5:
                break
        predictions[id1] = similar_docs
    
    return predictions

if __name__ == "__main__":
    predictions = main()
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import torch
from gensim.models import KeyedVectors
from tqdm import tqdm
from collections import defaultdict

# Load files
with open('ids.txt', 'r') as f:
    ids = f.read().splitlines()

with open('texts.txt', 'r') as f:
    texts = f.read().splitlines()

with open('items.json', 'r') as f:
    ground_truth = json.load(f)

# Step 1: TF-IDF Embeddings
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
tfidf_vectors = tfidf_vectorizer.fit_transform(texts)

# Step 2: GloVe Embeddings (load pre-trained GloVe model)
def load_glove_embeddings(file_path):
    glove_embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector
    return glove_embeddings

def get_glove_sentence_embedding(sentence, glove_model, dim=300):
    words = sentence.split()
    embeddings = [glove_model.get(word, np.zeros(dim)) for word in words]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(dim)

glove_model = load_glove_embeddings('glove.6B.300d.txt')
glove_vectors = np.array([get_glove_sentence_embedding(text, glove_model) for text in texts])

# Step 3: Word2Vec Embeddings (load pre-trained Word2Vec model)
word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
def get_word2vec_sentence_embedding(sentence, word2vec_model, dim=300):
    words = sentence.split()
    embeddings = [word2vec_model[word] if word in word2vec_model else np.zeros(dim) for word in words]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(dim)

word2vec_vectors = np.array([get_word2vec_sentence_embedding(text, word2vec_model) for text in texts])

# # Step 4: BERT Embeddings (using HuggingFace transformers)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# def get_bert_embedding(text, tokenizer, model):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# bert_vectors = np.array([get_bert_embedding(text, tokenizer, model) for text in texts])

# Step 5: Combine All Embeddings (Concatenation)
# combined_vectors = np.hstack([tfidf_vectors.toarray(), glove_vectors, word2vec_vectors, bert_vectors])
combined_vectors = np.hstack([tfidf_vectors.toarray(), glove_vectors, word2vec_vectors])

# Step 6: Apply LSH using Nearest Neighbors
n_neighbors = 5
lsh_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
lsh_model.fit(combined_vectors)

# Step 7: Predict top 5 similar items using LSH
predicted_map = defaultdict(list)
for idx in tqdm(range(combined_vectors.shape[0])):
    _, indices = lsh_model.kneighbors([combined_vectors[idx]], n_neighbors=n_neighbors+1)  # n+1 because self is also a neighbor
    predicted_indices = [ids[i] for i in indices.flatten() if i != idx][:5]  # Exclude self and limit to 5
    predicted_map[ids[idx]] = predicted_indices

# Step 8: Evaluate using Intersection Score
intersection_scores = []
for sample_id in ids:
    predicted_set = set(predicted_map[sample_id])
    true_set = set(ground_truth.get(sample_id, []))
    intersection_score = len(predicted_set.intersection(true_set))
    intersection_scores.append(intersection_score)

# Step 9: Statistics and Visualization
intersection_df = pd.DataFrame(intersection_scores, columns=['Intersection Score'])

# Describe statistics
print(intersection_df.describe())

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(intersection_scores, bins=6, edgecolor='black')
plt.title('Histogram of Intersection Scores')
plt.xlabel('Intersection Score')
plt.ylabel('Frequency')
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
plt.boxplot(intersection_scores)
plt.title('Box Plot of Intersection Scores')
plt.ylabel('Intersection Score')
plt.show()

# Step 10: Save predicted map to file
with open('predicted_items.json', 'w') as f:
    json.dump(predicted_map, f, indent=4)

