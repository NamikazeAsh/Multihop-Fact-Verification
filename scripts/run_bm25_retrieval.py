"""
Run BM25 retrieval on HoVer datasets
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.bm25_retriever.hover_project import BM25Retriever
import json
import os

print("="*60)
print("HoVer BM25 Retrieval")
print("="*60)

# Create output directory
os.makedirs('output', exist_ok=True)

# Initialize retriever
retriever = BM25Retriever()

# Process each dataset
datasets = {
    'train': 'data/hover_train_release_v1.1.json',
    'dev': 'data/hover_dev_release_v1.1.json',
    'test': 'data/hover_test_release_v1.1.json'
}

for split, filepath in datasets.items():
    print(f"\n{'='*60}")
    print(f"Processing {split.upper()} set")
    print(f"{'='*60}")
    
    # Load data
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    # Retrieve top-100 documents for each claim
    output_file = f'output/hover_{split}_bm25_top100.json'
    print(f"Retrieving top-100 documents per claim...")
    
    results = retriever.batch_retrieve(
        data, 
        k=100, 
        output_file=output_file
    )
    
    print(f"✓ Saved results to: {output_file}")
    
    # Evaluate retrieval quality (skip test set - no labels)
    if split != 'test':
        print(f"\nEvaluating retrieval quality...")
        metrics = retriever.evaluate_retrieval(results)
        
        print(f"\n📊 Retrieval Metrics for {split}:")
        print(f"  Total claims: {metrics['total_claims']}")
        print(f"  Claims with ALL supporting docs: {metrics['claims_with_all_docs']}")
        print(f"  Coverage: {metrics['coverage']:.2%}")
        print(f"  Average Recall: {metrics['avg_recall']:.2%}")

print(f"\n{'='*60}")
print("✓ Retrieval Complete!")
print(f"{'='*60}")
print("\nNext steps:")
print("1. Implement sentence selection from retrieved documents")
print("2. Build verification model (BERT/RoBERTa)")
print("3. Train and evaluate")