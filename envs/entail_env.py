#!/usr/bin/env python3
"""
Gym environment for case entailment/relationship classification task.
"""
from typing import Dict, Any, Optional
from .base_env import BaseCapRLVREnv


class EntailmentEnv(BaseCapRLVREnv):
    """
    Environment for legal case relationship classification task.
    
    Task: Given two legal cases, determine their relationship 
    (e.g., AFFIRMS, REVERSES, DISTINGUISHES, CITES, etc.).
    """
    
    def __init__(self, 
                 data_path: str = "data_tasks/entail/train.jsonl",
                 max_episode_length: int = 1,
                 subset_size: Optional[int] = None):
        """
        Initialize entailment environment.
        
        Args:
            data_path: Path to entail task JSONL file
            max_episode_length: Steps per episode (default: 1)
            subset_size: Limit dataset size for faster training
        """
        super().__init__(
            data_path=data_path,
            task_type='entail',
            max_episode_length=max_episode_length,
            subset_size=subset_size
        )
        
        # Define possible relationship labels
        self.relationship_labels = [
            'AFFIRMS', 'REVERSES', 'DISTINGUISHES', 'CITES', 
            'FOLLOWS', 'OVERRULES', 'CRITICIZES', 'QUESTIONS',
            'NEUTRAL', 'UNKNOWN'
        ]
    
    def _get_ground_truth(self) -> Any:
        """Get ground truth relationship for current sample"""
        if not self.current_sample:
            return None
        
        return {
            'label': self.current_sample.get('label', 'UNKNOWN'),
            'relationship': self.current_sample.get('relationship', ''),
            'citing_case': self.current_sample.get('citing_case', ''),
            'cited_case': self.current_sample.get('cited_case', ''),
            'context': self.current_sample.get('context', '')
        }
    
    def _extract_relationship_from_response(self, response: str) -> str:
        """Extract relationship label from model response"""
        response_upper = response.upper()
        
        # Look for explicit relationship labels
        for label in self.relationship_labels:
            if label in response_upper:
                return label
        
        # Look for common phrases that indicate relationships
        relationship_patterns = {
            'AFFIRMS': ['affirm', 'uphold', 'confirm', 'agree'],
            'REVERSES': ['reverse', 'overturn', 'reject', 'disagree'],
            'DISTINGUISHES': ['distinguish', 'different', 'separate'],
            'CITES': ['cite', 'reference', 'mention'],
            'FOLLOWS': ['follow', 'adopt', 'accept'],
            'OVERRULES': ['overrule', 'supersede', 'replace'],
            'CRITICIZES': ['criticize', 'disapprove', 'condemn'],
            'QUESTIONS': ['question', 'doubt', 'uncertain']
        }
        
        for label, patterns in relationship_patterns.items():
            for pattern in patterns:
                if pattern in response.lower():
                    return label
        
        return 'UNKNOWN'
    
    def render(self, mode='human') -> Optional[str]:
        """Render entailment environment"""
        if not self.current_sample:
            return "No current sample"
        
        output = f"=== Case Entailment Task ===\n"
        output += f"Pair ID: {self.current_sample.get('pair_id', 'unknown')}\n"
        output += f"Step: {self.current_step}/{self.max_episode_length}\n\n"
        
        # Show the relationship classification prompt
        output += f"Task: {self.current_sample.get('inputs', 'No input')}\n\n"
        
        # Show case information
        if 'citing_case' in self.current_sample:
            output += f"Citing Case: {self.current_sample['citing_case']}\n"
        
        if 'cited_case' in self.current_sample:
            output += f"Cited Case: {self.current_sample['cited_case']}\n"
        
        # Show context if available
        if 'context' in self.current_sample:
            context = self.current_sample['context'][:200]
            output += f"\nContext: {context}...\n"
        
        # Show possible relationships
        output += f"\nPossible relationships: {', '.join(self.relationship_labels)}\n"
        
        # Show correct answer if available
        if 'label' in self.current_sample:
            output += f"\nCorrect Relationship: {self.current_sample['label']}\n"
        
        if mode == 'human':
            print(output)
        else:
            return output
    
    def step(self, action: str) -> tuple:
        """
        Execute entailment step and evaluate relationship classification.
        
        The action should contain the predicted relationship.
        """
        # Extract relationship from response
        predicted_relationship = self._extract_relationship_from_response(action)
        
        # Add prediction to sample for reward computation
        if self.current_sample:
            self.current_sample['predicted_label'] = predicted_relationship
        
        # Call parent step method
        obs, reward, done, info = super().step(action)
        
        # Add entailment-specific info
        info['predicted_relationship'] = predicted_relationship
        info['available_labels'] = self.relationship_labels
        
        return obs, reward, done, info


def test_entail_env():
    """Test the entailment environment"""
    import os
    
    # Check if data file exists
    data_path = "/Users/kyle/Developer/cap-rlvr/data_tasks/entail/train.jsonl"
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Creating test environment with empty data...")
        data_path = "nonexistent.jsonl"  # Will create empty environment
    
    # Create environment with small subset for testing
    env = EntailmentEnv(data_path=data_path, subset_size=10)
    
    print(f"Created entailment environment with {env.get_sample_count()} samples")
    
    # Test reset and step
    obs = env.reset()
    print(f"Initial observation keys: {list(obs.keys())}")
    
    # Render the environment
    env.render()
    
    # Test a step with sample response
    test_response = """
    Based on the context provided, the citing case AFFIRMS the cited case.
    The court agreed with the previous decision and upheld the ruling.
    """
    
    next_obs, reward, done, info = env.step(test_response)
    
    print(f"\nStep results:")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Predicted relationship: {info.get('predicted_relationship')}")
    
    env.close()


if __name__ == '__main__':
    test_entail_env()