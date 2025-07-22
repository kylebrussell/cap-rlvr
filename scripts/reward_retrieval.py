#!/usr/bin/env python3
"""
Reward function for retrieval task using FAISS embeddings.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import re

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Install with: pip install faiss-cpu sentence-transformers")

class RetrievalRewardFunction:
    def __init__(self, faiss_index_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize retrieval reward function with pre-built FAISS index"""
        self.index_path = Path(faiss_index_path)
        self.metadata_path = self.index_path.with_suffix('.metadata.json')
        
        # Load FAISS index
        print(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        
        # Load metadata
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.case_ids = self.metadata['case_ids']
        self.id_to_index = self.metadata['id_to_index']
        
        # Load sentence transformer for encoding new queries
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print(f"Loaded index with {len(self.case_ids)} cases, dimension {self.index.d}")
    
    def extract_case_names_from_response(self, response: str) -> List[str]:
        """Extract case names/citations from model response"""
        # Look for various citation patterns
        citations = []
        
        # Pattern 1: Case v. Case format
        case_pattern = r'([A-Z][A-Za-z\s&]+(?:v\.|vs?\.|versus)\s+[A-Z][A-Za-z\s&]+)'
        matches = re.findall(case_pattern, response)
        citations.extend(matches)
        
        # Pattern 2: Volume Reporter Page (Year) format
        citation_pattern = r'(\d+\s+[A-Za-z\.]+\s+\d+\s*\(\d{4}\))'
        matches = re.findall(citation_pattern, response)
        citations.extend(matches)
        
        # Pattern 3: Look for lines starting with numbers (numbered lists)
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.?\s+', line):
                # Remove the number prefix
                clean_line = re.sub(r'^\d+\.?\s+', '', line).strip()
                if len(clean_line) > 10:  # Reasonable minimum length
                    citations.append(clean_line)
        
        # Clean and deduplicate
        cleaned_citations = []
        for citation in citations:
            citation = citation.strip()
            if citation and len(citation) > 5:  # Basic length filter
                cleaned_citations.append(citation)
        
        return list(set(cleaned_citations))  # Remove duplicates
    
    def compute_similarity_reward(self, query_facts: str, model_response: str, 
                                ground_truth_cases: List[str]) -> float:
        """Compute reward based on similarity between retrieved and ground truth cases"""
        # Extract case names from model response
        suggested_cases = self.extract_case_names_from_response(model_response)
        
        if not suggested_cases:
            return 0.0  # No valid cases suggested
        
        # Encode query facts
        query_embedding = self.model.encode([query_facts], convert_to_numpy=True)
        
        # Find similar cases for both suggested and ground truth
        k = min(20, len(self.case_ids))  # Search top-k similar cases
        
        # Get embeddings for ground truth cases (if they exist in our index)
        gt_indices = []
        for gt_case in ground_truth_cases:
            if gt_case in self.id_to_index:
                gt_indices.append(self.id_to_index[gt_case])
        
        if not gt_indices:
            # Fallback: use similarity search for ground truth text
            gt_similarities = []
            for gt_case in ground_truth_cases:
                gt_embedding = self.model.encode([str(gt_case)], convert_to_numpy=True)
                _, indices = self.index.search(gt_embedding.astype(np.float32), 1)
                if indices[0][0] < len(self.case_ids):
                    gt_similarities.append(self.case_ids[indices[0][0]])
            
            if not gt_similarities:
                return 0.1  # Small reward for attempting retrieval
        
        # Compute similarities for suggested cases
        suggested_similarities = []
        for suggested in suggested_cases[:5]:  # Limit to top 5 suggestions
            suggested_embedding = self.model.encode([suggested], convert_to_numpy=True)
            distances, indices = self.index.search(suggested_embedding.astype(np.float32), k)
            
            # Convert distances to similarities (lower distance = higher similarity)
            similarities = 1.0 / (1.0 + distances[0])
            suggested_similarities.append(similarities)
        
        # Compute ground truth similarities
        gt_similarities = []
        for gt_idx in gt_indices[:5]:
            if gt_idx < len(self.case_ids):
                # Get the embedding for this case and find similar cases
                gt_embedding = self.model.encode([self.case_ids[gt_idx]], convert_to_numpy=True)
                distances, indices = self.index.search(gt_embedding.astype(np.float32), k)
                similarities = 1.0 / (1.0 + distances[0])
                gt_similarities.append(similarities)
        
        if not gt_similarities:
            return 0.1
        
        # Compare suggested vs ground truth similarity patterns
        # Reward based on how similar the suggested cases are to ground truth cases
        max_overlap_score = 0.0
        
        for sugg_sims in suggested_similarities:
            for gt_sims in gt_similarities:
                # Compute correlation or overlap between similarity patterns
                overlap = np.corrcoef(sugg_sims[:min(len(sugg_sims), len(gt_sims))], 
                                    gt_sims[:min(len(sugg_sims), len(gt_sims))])[0, 1]
                if not np.isnan(overlap):
                    max_overlap_score = max(max_overlap_score, overlap)
        
        # Normalize to 0-1 range
        reward = max(0.0, min(1.0, (max_overlap_score + 1.0) / 2.0))
        
        return reward
    
    def reward(self, sample: Dict, model_output: str) -> float:
        """
        Main reward function for retrieval task.
        
        Args:
            sample: Dictionary with 'inputs' (query facts) and 'positives' (ground truth cases)
            model_output: Model's response with suggested similar cases
            
        Returns:
            Float reward between 0.0 and 1.0
        """
        try:
            # Extract query facts from inputs
            query_facts = sample['inputs'].replace('Find 5 cases with similar legal issues and facts:\n\nFacts: ', '')
            
            # Get ground truth cases
            ground_truth_cases = sample.get('positives', [])
            
            if not ground_truth_cases:
                # If no ground truth, give small reward for any reasonable attempt
                suggested_cases = self.extract_case_names_from_response(model_output)
                return 0.3 if suggested_cases else 0.0
            
            # Compute similarity-based reward
            similarity_reward = self.compute_similarity_reward(
                query_facts, model_output, ground_truth_cases
            )
            
            # Bonus for extracting reasonable number of cases
            suggested_cases = self.extract_case_names_from_response(model_output)
            quantity_bonus = min(0.2, len(suggested_cases) * 0.04)  # Up to 0.2 bonus for 5+ cases
            
            total_reward = similarity_reward + quantity_bonus
            return min(1.0, total_reward)
            
        except Exception as e:
            print(f"Error computing retrieval reward: {e}")
            return 0.0

# Example usage function
def test_reward_function():
    """Test the retrieval reward function"""
    # This would be called after building the FAISS index
    reward_fn = RetrievalRewardFunction('data_tasks/retrieval/embeddings.faiss')
    
    # Example test case
    sample = {
        'inputs': 'Find 5 cases with similar legal issues and facts:\n\nFacts: Plaintiff sued for breach of contract involving software licensing agreement...',
        'positives': ['Smith v. Microsoft Corp.', 'Oracle Corp. v. Johnson']
    }
    
    model_output = """
    1. Smith v. Microsoft Corp. - Similar software licensing dispute
    2. Adobe Systems v. Thompson - Contract breach in software context
    3. Oracle Corp. v. Johnson - Licensing agreement violation
    4. TechCorp v. StartupInc - Breach of software development contract
    5. Microsoft v. DataSoft - Software licensing terms dispute
    """
    
    reward = reward_fn.reward(sample, model_output)
    print(f"Test reward: {reward:.3f}")

if __name__ == '__main__':
    test_reward_function()