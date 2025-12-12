"""
Run this to index Wikipedia into Elasticsearch
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.bm25_retriever.hover_project import WikipediaIndexer
import os

print("Starting Wikipedia indexing...")
print("This will take 30-60 minutes. Go grab a coffee! ☕")

# Initialize indexer
indexer = WikipediaIndexer()

# Create the index
print("\n[Step 1/2] Creating Elasticsearch index...")
indexer.create_index()

# Index Wikipedia
wiki_dir = 'data/enwiki-20171001-pages-meta-current-withlinks-processed/'
if os.path.exists(wiki_dir):
    print(f"\n[Step 2/2] Indexing Wikipedia from {wiki_dir}...")
    indexer.index_wikipedia(wiki_dir)
    print("\n✓ Indexing complete!")
else:
    print(f"\n✗ Wikipedia directory not found: {wiki_dir}")
    print("Make sure the path is correct relative to this script")