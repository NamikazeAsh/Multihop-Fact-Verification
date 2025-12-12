import json
import os
import bz2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm
import subprocess # For executing shell commands

def run_shell_command(command, description="Executing shell command"):
    """Helper to run shell commands and print output."""
    print(f"\n--- {description} ---")
    try:
        # Use subprocess.run for better control and error handling
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise # Re-raise to stop execution if a critical command fails
    print(f"--- Finished {description} ---")

def load_wikipedia_articles(data_path: str):
    """
    Loads Wikipedia articles from a specified path.
    Handles both directories with .bz2 files and single decompressed files.
    """
    articles = []
    print(f"Loading Wikipedia articles from {data_path}...")
    
    files_to_process = []
    
    # Determine if data_path is a single file or a directory
    if os.path.isfile(data_path):
        files_to_process = [data_path]
    elif os.path.isdir(data_path):
        # If it's a directory, look for the actual data file(s) inside it.
        # This covers both the local setup (directory of .bz2 chunks)
        # and the Colab setup (directory containing a single large decompressed file).
        
        # Look for the single large decompressed file first (expected from Colab bunzip2 output)
        decompressed_file_name = "enwiki-20171001-pages-meta-current-withlinks-processed"
        potential_single_file = os.path.join(data_path, decompressed_file_name)
        
        if os.path.isfile(potential_single_file):
            files_to_process.append(potential_single_file)
        else:
            # If not a single decompressed file directly, look for .bz2 chunks in subdirectories
            for root, _, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.bz2'):
                        files_to_process.append(os.path.join(root, file))
            
    files_to_process.sort() # Ensure consistent order
    if not files_to_process:
        print(f"Warning: No data files (.bz2 or decompressed) found in {data_path}.")
        return []

    print(f"Found {len(files_to_process)} data files to process.")

    for file_path in tqdm(files_to_process, desc="Processing data files"):
        is_bz2 = file_path.endswith('.bz2')
        
        try:
            # Open file with bz2.open if compressed, else open normally
            with (bz2.open(file_path, 'rt', encoding='utf-8') if is_bz2 else open(file_path, 'r', encoding='utf-8')) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        
                        text_data = doc.get('text', [])
                        sentences = []
                        for item in text_data:
                            if isinstance(item, list):
                                sentences.extend([s for s in item if isinstance(s, str)])
                            elif isinstance(item, str):
                                sentences.append(item)
                        
                        full_text = ' '.join(sentences)
                        
                        articles.append({
                            "id": doc.get('id', doc['title']), # Use ID if available, else title
                            "title": doc['title'],
                            "text": full_text
                        })
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        pass
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
    print(f"Loaded {len(articles)} Wikipedia articles.")
    return articles

if __name__ == "__main__":
    print("--- Starting Dense Retrieval Setup (Local) ---")

    # Path to the Wikipedia data for loading
    # Assumes data is already extracted in this folder
    WIKI_DATA_PATH_FOR_LOADING = "data/enwiki-20171001-pages-meta-current-withlinks-processed/"

    if not os.path.exists(WIKI_DATA_PATH_FOR_LOADING):
        print(f"Error: Data directory not found at {WIKI_DATA_PATH_FOR_LOADING}")
        print("Please ensure you have extracted the Wikipedia dump into this folder.")
        exit(1)

    # Load Wikipedia Data
    wikipedia_articles = load_wikipedia_articles(WIKI_DATA_PATH_FOR_LOADING)

    if wikipedia_articles:
        print("\nSample Article:")
        print(json.dumps(wikipedia_articles[0], indent=2))
    else:
        print("\nNo Wikipedia articles loaded. Cannot proceed with embedding and indexing.")
        exit()

    # Load Sentence-BERT Model and Generate Embeddings
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    print(f"Loaded Sentence-BERT model: {model_name}")

    corpus_texts = [article['text'] for article in wikipedia_articles]
    corpus_ids = [article['id'] for article in wikipedia_articles]

    print(f"Generating embeddings for {len(corpus_texts)} Wikipedia articles...")

    chunk_size = 1000 
    all_embeddings = []

    for i in tqdm(range(0, len(corpus_texts), chunk_size), desc="Generating Embeddings"):
        chunk = corpus_texts[i:i + chunk_size]
        chunk_embeddings = model.encode(chunk, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(chunk_embeddings)

    if all_embeddings:
        corpus_embeddings = np.vstack(all_embeddings)
        print(f"Generated embeddings with shape: {corpus_embeddings.shape}")
    else:
        print("No embeddings generated! Exiting.")
        exit()

    # Build FAISS Index
    if corpus_embeddings.shape[0] > 0:
        embedding_dim = corpus_embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(corpus_embeddings)
        print(f"FAISS index created with {index.ntotal} vectors.")
    else:
        print("No embeddings to build FAISS index. Exiting.")
        exit()

    # Save FAISS Index and Embeddings
    os.makedirs("dense_retrieval_artifacts", exist_ok=True)

    faiss_index_path = "dense_retrieval_artifacts/wikipedia_faiss_index.bin"
    faiss.write_index(index, faiss_index_path)
    print(f"FAISS index saved to {faiss_index_path}")

    corpus_ids_path = "dense_retrieval_artifacts/wikipedia_corpus_ids.json"
    with open(corpus_ids_path, 'w', encoding='utf-8') as f:
        json.dump(corpus_ids, f)
    print(f"Corpus IDs saved to {corpus_ids_path}")

    corpus_embeddings_path = "dense_retrieval_artifacts/wikipedia_corpus_embeddings.npy"
    np.save(corpus_embeddings_path, corpus_embeddings)
    print(f"Corpus embeddings saved to {corpus_embeddings_path}")

    # Test Retrieval
    print("\n--- Testing Retrieval ---")
    query_text = "Who invented the light bulb?"
    query_embedding = model.encode(query_text, convert_to_numpy=True)
    query_embedding = np.array([query_embedding])

    k_neighbors = 5
    distances, indices = index.search(query_embedding, k_neighbors)

    print(f"\nTop {k_neighbors} most similar articles for query: '{query_text}'")
    for i, idx in enumerate(indices[0]):
        if idx < len(corpus_ids):
            article_id = corpus_ids[idx]
            article_title = next((a['title'] for a in wikipedia_articles if a['id'] == article_id), "Unknown")
            distance = distances[0][i]
            print(f"Rank {i+1}: Title='{article_title}' (ID: {article_id}) - Distance: {distance:.4f}")
        else:
            print(f"Warning: Index {idx} out of bounds for corpus_ids.")

    print("\n--- Dense Retrieval Setup Complete ---")
