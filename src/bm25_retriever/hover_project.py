"""
HoVer Fact Verification Project - Setup and BM25 Retrieval
Uses HotpotQA Wikipedia dump for document retrieval
"""

import json
import bz2
from elasticsearch import Elasticsearch, helpers
from typing import List, Dict, Tuple
from tqdm import tqdm
import os

class WikipediaIndexer:
    """
    Index HotpotQA Wikipedia dump into Elasticsearch for BM25 retrieval.
    """
    
    def __init__(self, host='localhost', port=9200):
        """Initialize Elasticsearch connection."""
        self.es = Elasticsearch([f'http://{host}:{port}'])
        self.index_name = 'hotpot_wiki'
        
    def create_index(self):
        """Create Elasticsearch index optimized for BM25 retrieval."""
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard"
                        }
                    }
                },
                "similarity": {
                    "bm25_similarity": {
                        "type": "BM25",
                        "k1": 1.2,  # Controls term frequency saturation
                        "b": 0.75   # Controls length normalization
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "similarity": "bm25_similarity",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "text": {
                        "type": "text",
                        "similarity": "bm25_similarity"
                    },
                    "sentences": {
                        "type": "text",
                        "similarity": "bm25_similarity"
                    }
                }
            }
        }
        
        if self.es.indices.exists(index=self.index_name):
            print(f"Index {self.index_name} already exists. Using existing Index!!")
            #self.es.indices.delete(index=self.index_name)
            #raise ValueError("Index already exists!")
            
        self.es.indices.create(index=self.index_name, body=index_settings)
        print(f"Created index: {self.index_name}")
        
    def parse_hotpot_wiki(self, wiki_dir_path: str):
        """
        Parse HotpotQA Wikipedia dump from directory structure.
        
        Expected structure:
        wiki_dir_path/
          ├── AA/
          │   └── wiki_00.bz2
          ├── AB/
          │   └── wiki_00.bz2
          └── ...
        
        Each .bz2 file contains JSON objects, one per line:
        {
            "id": "...",
            "url": "...",
            "title": "...",
            "text": ["sentence1", "sentence2", ...]
        }
        """
        print(f"Reading Wikipedia dump from {wiki_dir_path}")
        
        # Check if path is a directory or single file
        if os.path.isfile(wiki_dir_path):
            # Single file case
            files_to_process = [wiki_dir_path]
        else:
            # Directory case - find all .bz2 files
            files_to_process = []
            for root, dirs, files in os.walk(wiki_dir_path):
                for file in files:
                    if file.endswith('.bz2'):
                        files_to_process.append(os.path.join(root, file))
            
            files_to_process.sort()
            print(f"Found {len(files_to_process)} .bz2 files to process")
        
        # Process each file
        for file_path in files_to_process:            
            try:
                with bz2.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            doc = json.loads(line)
                            yield doc
                        except json.JSONDecodeError as e:
                            # Skip malformed lines
                            continue
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    def index_wikipedia(self, wiki_dir_path: str, batch_size=1000):
        """
        Index Wikipedia documents into Elasticsearch.
        
        Args:
            wiki_dir_path: Path to HotpotQA wiki directory or file
            batch_size: Number of documents to index at once
        """
        def generate_docs():
            """Generator for bulk indexing."""
            for doc in self.parse_hotpot_wiki(wiki_dir_path):
                # Handle nested list structure in text field
                text_data = doc.get('text', [])
                
                # Flatten if nested lists (some docs have [[sent1], [sent2]])
                sentences = []
                for item in text_data:
                    if isinstance(item, list):
                        sentences.extend([s for s in item if isinstance(s, str)])
                    elif isinstance(item, str):
                        sentences.append(item)
                
                # Combine all sentences
                full_text = ' '.join(sentences)
                sentences_text = '\n'.join(sentences)
                
                yield {
                    "_index": self.index_name,
                    "_id": doc.get('id', doc['title']),  # Use ID if available
                    "_source": {
                        "title": doc['title'],
                        "text": full_text,
                        "sentences": sentences_text,
                        "url": doc.get('url', '')
                    }
                }
        
        print("Starting bulk indexing...")
        success_count = 0
        
        for ok, result in helpers.streaming_bulk(
            self.es,
            generate_docs(),
            chunk_size=batch_size,
            max_retries=3,
            request_timeout=60
        ):
            if ok:
                success_count += 1
                if success_count % 10000 == 0:
                    print(f"Indexed {success_count} documents...")
            else:
                print(f"Failed to index document: {result}")
        
        print(f"Successfully indexed {success_count} documents")
        self.es.indices.refresh(index=self.index_name)


class BM25Retriever:
    """BM25-based retrieval for HoVer claims."""
    
    def __init__(self, host='localhost', port=9200, index_name='hotpot_wiki'):
        self.es = Elasticsearch([f'http://{host}:{port}'])
        self.index_name = index_name
        
    def retrieve(self, claim: str, k: int = 100) -> List[Dict]:
        """
        Retrieve top-k documents for a claim using BM25.
        
        Args:
            claim: The claim text
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        query = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": claim,
                    "fields": ["title^3", "text^1"],  # Boost title 3x
                    "type": "best_fields"
                }
            },
            "_source": ["title", "sentences", "url"]
        }
        
        response = self.es.search(index=self.index_name, body=query)
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            results.append({
                'doc_id': hit['_id'],
                'title': source['title'],
                'sentences': source['sentences'].split('\n'),
                'score': hit['_score'],
                'url': source.get('url', '')
            })
            
        return results
    
    def batch_retrieve(self, dataset: List[Dict], k: int = 100, 
                      output_file: str = None) -> Dict[str, List[Dict]]:
        """
        Retrieve documents for all claims in dataset.
        
        Args:
            dataset: List of HoVer examples
            k: Number of documents per claim
            output_file: Optional path to save results
            
        Returns:
            Dictionary mapping UIDs to retrieved documents
        """
        results = {}
        
        for example in tqdm(dataset, desc="Retrieving documents"):
            uid = example['uid']
            claim = example['claim']
            
            retrieved_docs = self.retrieve(claim, k=k)
            results[uid] = {
                'claim': claim,
                'retrieved_docs': retrieved_docs,
                'label': example.get('label', None),
                'supporting_facts': example.get('supporting_facts', [])
            }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved results to {output_file}")
            
        return results
    
    def evaluate_retrieval(self, results: Dict[str, Dict]) -> Dict:
        """
        Evaluate retrieval performance by checking if supporting facts
        are in retrieved documents.
        
        Args:
            results: Retrieval results with supporting_facts
            
        Returns:
            Dictionary with evaluation metrics
        """
        total_claims = len(results)
        claims_with_all_docs = 0
        total_recall = 0
        
        for uid, data in results.items():
            supporting_facts = data['supporting_facts']
            retrieved_titles = {doc['title'] for doc in data['retrieved_docs']}
            
            # Extract unique titles from supporting facts
            required_titles = {fact[0] for fact in supporting_facts}
            
            # Check recall
            found_titles = required_titles.intersection(retrieved_titles)
            recall = len(found_titles) / len(required_titles) if required_titles else 0
            total_recall += recall
            
            if recall == 1.0:
                claims_with_all_docs += 1
        
        return {
            'total_claims': total_claims,
            'claims_with_all_docs': claims_with_all_docs,
            'coverage': claims_with_all_docs / total_claims,
            'avg_recall': total_recall / total_claims
        }


def setup_project():
    """Complete project setup workflow."""
    
    print("=" * 60)
    print("HoVer Fact Verification Project Setup")
    print("=" * 60)
    
    # Step 1: Load HoVer datasets
    print("\n[Step 1] Loading HoVer datasets...")
    
    data_paths = {
        'train': 'hover_train.json',
        'dev': 'hover_dev.json',
        'test': 'hover_test.json'
    }
    
    datasets = {}
    for split, path in data_paths.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                datasets[split] = json.load(f)
            print(f"  Loaded {split}: {len(datasets[split])} examples")
        else:
            print(f"  Warning: {path} not found")
    
    # Step 2: Setup Elasticsearch indexing
    print("\n[Step 2] Setting up Elasticsearch...")
    print("  Make sure Elasticsearch is running on localhost:9200")
    print("  Start with: docker run -p 9200:9200 -e 'discovery.type=single-node' elasticsearch:7.17.0")
    
    indexer = WikipediaIndexer()
    
    # Create index
    indexer.create_index()
    
    # Step 3: Index Wikipedia (only if you have the file)
    wiki_file = 'enwiki-20171001-pages-meta-current-withlinks-abstracts'
    
    if os.path.exists(wiki_file) or os.path.exists(f"{wiki_file}.bz2"):
        print(f"\n[Step 3] Indexing Wikipedia from {wiki_file}...")
        print("  This will take 30-60 minutes for full Wikipedia...")
        
        wiki_path = f"{wiki_file}.bz2" if os.path.exists(f"{wiki_file}.bz2") else wiki_file
        indexer.index_wikipedia(wiki_path)
    else:
        print(f"\n[Step 3] Wikipedia file not found: {wiki_file}")
        print("  Download from: https://hotpotqa.github.io/wiki-readme.html")
        print("  Or extract from: enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2")
        return
    
    # Step 4: Run retrieval
    print("\n[Step 4] Running BM25 retrieval...")
    retriever = BM25Retriever()
    
    for split, data in datasets.items():
        print(f"\n  Processing {split} set...")
        output_file = f"hover_{split}_bm25_top100.json"
        
        results = retriever.batch_retrieve(data, k=100, output_file=output_file)
        
        # Evaluate
        if split != 'test':  # Test set doesn't have labels
            metrics = retriever.evaluate_retrieval(results)
            print(f"\n  Retrieval Evaluation for {split}:")
            print(f"    Coverage (all docs found): {metrics['coverage']:.2%}")
            print(f"    Average Recall: {metrics['avg_recall']:.2%}")
    
    print("\n" + "=" * 60)
    print("Setup complete! Next steps:")
    print("1. Implement sentence selection from retrieved documents")
    print("2. Build verification model (BERT/RoBERTa)")
    print("3. Train and evaluate on HoVer dataset")
    print("=" * 60)


if __name__ == "__main__":
    # Quick test without full setup
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full-setup':
        setup_project()
    else:
        print("HoVer BM25 Retrieval Project")
        print("\nUsage:")
        print("  python hover_project.py --full-setup    # Run complete setup")
        print("\nOr use components individually:")
        print("  1. indexer = WikipediaIndexer()")
        print("  2. indexer.create_index()")
        print("  3. indexer.index_wikipedia('wiki_file.bz2')")
        print("  4. retriever = BM25Retriever()")
        print("  5. results = retriever.retrieve('your claim', k=100)")