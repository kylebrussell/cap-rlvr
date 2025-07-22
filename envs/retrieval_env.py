#!/usr/bin/env python3
"""
Gym environment for case retrieval task.
"""
from typing import Dict, Any, Optional, List
from .base_env import BaseCapRLVREnv


class CaseRetrievalEnv(BaseCapRLVREnv):
    """
    Environment for analogous case retrieval task.
    
    Task: Given a legal scenario, find and retrieve similar/analogous cases
    from the case database.
    """
    
    def __init__(self, 
                 data_path: str = "data_tasks/retrieval/train.jsonl",
                 faiss_index_path: Optional[str] = "data_tasks/retrieval/embeddings.faiss",
                 max_episode_length: int = 1,
                 subset_size: Optional[int] = None):
        """
        Initialize case retrieval environment.
        
        Args:
            data_path: Path to retrieval task JSONL file
            faiss_index_path: Path to FAISS index for similarity search
            max_episode_length: Steps per episode (default: 1)
            subset_size: Limit dataset size for faster training
        """
        super().__init__(
            data_path=data_path,
            task_type='retrieval',
            max_episode_length=max_episode_length,
            faiss_index_path=faiss_index_path,
            subset_size=subset_size
        )
    
    def _get_ground_truth(self) -> Any:
        """Get ground truth cases for current sample"""
        if not self.current_sample:
            return None
        
        return {
            'positives': self.current_sample.get('positives', []),
            'related_cases': self.current_sample.get('related_cases', []),
            'query_case': self.current_sample.get('query_case', ''),
            'expected_count': len(self.current_sample.get('positives', []))
        }
    
    def _extract_case_ids_from_response(self, response: str) -> List[str]:
        """Extract case IDs from model response"""
        import re
        
        # Look for common case ID patterns
        patterns = [
            r'\b\d{4,7}-\d+\b',  # Standard case ID format
            r'\b[A-Z]{2,3}_\d+\b',  # State abbreviation + number
            r'\b\d{4,7}\b',  # Simple numeric IDs
        ]
        
        case_ids = []
        for pattern in patterns:
            matches = re.findall(pattern, response)
            case_ids.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_case_ids = []
        for case_id in case_ids:
            if case_id not in seen:
                seen.add(case_id)
                unique_case_ids.append(case_id)
        
        return unique_case_ids[:10]  # Limit to top 10 retrieved cases
    
    def render(self, mode='human') -> Optional[str]:
        """Render case retrieval environment"""
        if not self.current_sample:
            return "No current sample"
        
        output = f"=== Case Retrieval Task ===\n"
        output += f"Query ID: {self.current_sample.get('query_case', 'unknown')}\n"
        output += f"Step: {self.current_step}/{self.max_episode_length}\n\n"
        
        # Show the retrieval prompt
        output += f"Query: {self.current_sample.get('inputs', 'No input')}\n\n"
        
        # Show expected number of cases to find
        if 'positives' in self.current_sample:
            num_expected = len(self.current_sample['positives'])
            output += f"Expected to find: {num_expected} similar cases\n"
        
        # Show some positive examples if available
        if 'positives' in self.current_sample and self.current_sample['positives']:
            output += "\nSome target cases:\n"
            for i, case_id in enumerate(self.current_sample['positives'][:3]):
                output += f"  {i+1}. {case_id}\n"
            
            if len(self.current_sample['positives']) > 3:
                output += f"  ... and {len(self.current_sample['positives'])-3} more\n"
        
        if mode == 'human':
            print(output)
        else:
            return output
    
    def step(self, action: str) -> tuple:
        """
        Execute retrieval step and evaluate retrieved cases.
        
        The action should contain case IDs of retrieved cases.
        """
        # Extract case IDs from the response
        retrieved_cases = self._extract_case_ids_from_response(action)
        
        # Add retrieved cases to info for reward computation
        if self.current_sample:
            self.current_sample['retrieved_cases'] = retrieved_cases
        
        # Call parent step method
        obs, reward, done, info = super().step(action)
        
        # Add retrieval-specific info
        info['retrieved_cases'] = retrieved_cases
        info['num_retrieved'] = len(retrieved_cases)
        
        return obs, reward, done, info


def test_retrieval_env():
    """Test the case retrieval environment"""
    import os
    
    # Check if data file exists
    data_path = "/Users/kyle/Developer/cap-rlvr/data_tasks/retrieval/train.jsonl"
    faiss_path = "/Users/kyle/Developer/cap-rlvr/data_tasks/retrieval/embeddings.faiss"
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Creating test environment with empty data...")
        data_path = "nonexistent.jsonl"  # Will create empty environment
    
    if not os.path.exists(faiss_path):
        print(f"FAISS index not found at {faiss_path}")
        faiss_path = None
    
    # Create environment with small subset for testing
    env = CaseRetrievalEnv(
        data_path=data_path, 
        faiss_index_path=faiss_path,
        subset_size=10
    )
    
    print(f"Created retrieval environment with {env.get_sample_count()} samples")
    
    # Test reset and step
    obs = env.reset()
    print(f"Initial observation keys: {list(obs.keys())}")
    
    # Render the environment
    env.render()
    
    # Test a step with sample response containing case IDs
    test_response = """
    Based on the query, I found the following similar cases:
    
    1. Case ID: 12345-67 - Similar contract dispute
    2. Case ID: 23456-78 - Analogous breach of contract
    3. Case ID: 34567-89 - Related commercial law case
    4. Case ID: 45678-90 - Comparable damages calculation
    5. Case ID: 56789-01 - Similar legal precedent
    """
    
    next_obs, reward, done, info = env.step(test_response)
    
    print(f"\nStep results:")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Retrieved cases: {info.get('retrieved_cases', [])}")
    print(f"Number retrieved: {info.get('num_retrieved', 0)}")
    
    env.close()


if __name__ == '__main__':
    test_retrieval_env()