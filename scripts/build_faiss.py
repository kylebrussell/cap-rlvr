#!/usr/bin/env python3
"""
Build FAISS index for retrieval task evaluation.
Creates frozen embeddings for efficient similarity search during training.
"""
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

try:
    import faiss
except ImportError:
    print("Error: faiss-cpu not installed. Run: pip install faiss-cpu")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed. Run: pip install sentence-transformers")
    exit(1)

def load_retrieval_data(jsonl_path: str) -> List[Dict]:
    """Load retrieval training data"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_case_texts(data: List[Dict]) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Extract all unique case texts and create ID mappings"""
    case_texts = []
    case_ids = []
    id_to_index = {}
    
    # Collect all unique cases (query cases + related cases)
    unique_cases = {}
    
    for item in data:
        # Add query case
        query_id = item['case_id']
        if query_id not in unique_cases:
            # Extract case facts as the text to embed
            case_text = item['inputs'].replace('Find 5 cases with similar legal issues and facts:\n\nFacts: ', '')
            unique_cases[query_id] = case_text
        
        # Add related cases (if we have their texts)
        for related_case in item.get('related_cases', []):
            rel_id = related_case['case_id']
            if rel_id not in unique_cases and 'case_text' in related_case:
                unique_cases[rel_id] = related_case['case_text']
    
    # Convert to lists for indexing
    for idx, (case_id, case_text) in enumerate(unique_cases.items()):
        case_ids.append(case_id)
        case_texts.append(case_text)
        id_to_index[case_id] = idx
    
    return case_texts, case_ids, id_to_index

def create_embeddings(case_texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Generate embeddings for all case texts"""
    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(case_texts)} cases...")
    
    # Process in batches to manage memory
    batch_size = 64
    embeddings = []
    
    for i in tqdm(range(0, len(case_texts), batch_size), desc="Encoding batches"):
        batch = case_texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    # Combine all embeddings
    all_embeddings = np.vstack(embeddings)
    print(f"Generated embeddings shape: {all_embeddings.shape}")
    
    return all_embeddings

def build_faiss_index(embeddings: np.ndarray, index_type: str = 'flat') -> faiss.Index:
    """Build FAISS index for similarity search"""
    dimension = embeddings.shape[1]
    print(f"Building FAISS index (dimension: {dimension}, type: {index_type})")
    
    if index_type == 'flat':
        # Exact search with L2 distance
        index = faiss.IndexFlatL2(dimension)
    elif index_type == 'ivf':
        # Approximate search with IVF (faster for large datasets)
        nlist = min(100, embeddings.shape[0] // 10)  # Number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        # Need to train the index
        print("Training IVF index...")
        index.train(embeddings)
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Add embeddings to index
    print(f"Adding {embeddings.shape[0]} vectors to index...")
    index.add(embeddings.astype(np.float32))
    
    return index

def save_index_and_metadata(index: faiss.Index, case_ids: List[str], 
                          id_to_index: Dict[str, int], output_path: str):
    """Save FAISS index and metadata"""
    output_path = Path(output_path)
    
    # Save FAISS index
    print(f"Saving FAISS index to {output_path}")
    faiss.write_index(index, str(output_path))
    
    # Save metadata
    metadata_path = output_path.with_suffix('.metadata.json')
    metadata = {
        'case_ids': case_ids,
        'id_to_index': id_to_index,
        'num_cases': len(case_ids),
        'index_type': type(index).__name__,
        'dimension': index.d
    }
    
    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Index build complete!")
    print(f"  - Cases indexed: {len(case_ids)}")
    print(f"  - Vector dimension: {index.d}")
    print(f"  - Index file: {output_path}")
    print(f"  - Metadata file: {metadata_path}")

def test_index(index: faiss.Index, case_ids: List[str], embeddings: np.ndarray, k: int = 5):
    """Test the index with a sample query"""
    if len(case_ids) == 0:
        print("No cases to test with")
        return
    
    print(f"\nTesting index with sample query (k={k})...")
    
    # Use first case as test query
    query_embedding = embeddings[0:1].astype(np.float32)
    distances, indices = index.search(query_embedding, k)
    
    print(f"Query case: {case_ids[0]}")
    print("Top similar cases:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx < len(case_ids):
            print(f"  {i+1}. {case_ids[idx]} (distance: {dist:.4f})")
        else:
            print(f"  {i+1}. Invalid index {idx}")

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for retrieval task")
    parser.add_argument('--in', '--input', dest='input_file', required=True,
                       help='Input JSONL file (e.g., data_tasks/retrieval/train.jsonl)')
    parser.add_argument('--out', '--output', dest='output_file', required=True,
                       help='Output FAISS index file (e.g., data_tasks/retrieval/embeddings.faiss)')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                       help='Sentence transformer model name (default: all-MiniLM-L6-v2)')
    parser.add_argument('--index-type', choices=['flat', 'ivf'], default='flat',
                       help='FAISS index type: flat (exact) or ivf (approximate)')
    parser.add_argument('--test', action='store_true',
                       help='Test the index after building')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading retrieval data from {args.input_file}")
    data = load_retrieval_data(args.input_file)
    print(f"Loaded {len(data)} retrieval examples")
    
    # Extract case texts
    case_texts, case_ids, id_to_index = extract_case_texts(data)
    print(f"Extracted {len(case_texts)} unique cases")
    
    if len(case_texts) == 0:
        print("Error: No case texts found in data")
        return
    
    # Generate embeddings
    embeddings = create_embeddings(case_texts, args.model)
    
    # Build FAISS index
    index = build_faiss_index(embeddings, args.index_type)
    
    # Save index and metadata
    save_index_and_metadata(index, case_ids, id_to_index, args.output_file)
    
    # Optional testing
    if args.test:
        test_index(index, case_ids, embeddings)

if __name__ == '__main__':
    main()