#!/usr/bin/env python3
"""
Gym environment for Bluebook citation completion task.
"""
from typing import Dict, Any, Optional
from .base_env import BaseCapRLVREnv


class BluebookCitationEnv(BaseCapRLVREnv):
    """
    Environment for legal citation completion task.
    
    Task: Given a partial legal citation, complete it in proper Bluebook format.
    """
    
    def __init__(self, 
                 data_path: str = "data_tasks/bluebook/train.jsonl",
                 max_episode_length: int = 1,
                 subset_size: Optional[int] = None):
        """
        Initialize Bluebook citation environment.
        
        Args:
            data_path: Path to bluebook task JSONL file
            max_episode_length: Steps per episode (default: 1)
            subset_size: Limit dataset size for faster training
        """
        super().__init__(
            data_path=data_path,
            task_type='bluebook',
            max_episode_length=max_episode_length,
            subset_size=subset_size
        )
    
    def _get_ground_truth(self) -> Any:
        """Get ground truth citation for current sample"""
        if not self.current_sample:
            return None
        
        return {
            'citation': self.current_sample.get('ground_truth', ''),
            'metadata': self.current_sample.get('metadata', {}),
            'case_name': self.current_sample.get('case_name', ''),
            'year': self.current_sample.get('year', '')
        }
    
    def render(self, mode='human') -> Optional[str]:
        """Render Bluebook citation environment"""
        if not self.current_sample:
            return "No current sample"
        
        output = f"=== Bluebook Citation Task ===\n"
        output += f"Case ID: {self.current_sample.get('case_id', 'unknown')}\n"
        output += f"Step: {self.current_step}/{self.max_episode_length}\n\n"
        
        # Show the citation prompt
        output += f"Complete the citation: {self.current_sample.get('inputs', 'No input')}\n\n"
        
        # Show metadata if available
        if 'metadata' in self.current_sample:
            metadata = self.current_sample['metadata']
            output += "Available metadata:\n"
            for key, value in metadata.items():
                output += f"  {key}: {value}\n"
        
        # Show correct answer if available
        if 'ground_truth' in self.current_sample:
            output += f"\nCorrect Citation: {self.current_sample['ground_truth']}\n"
        
        if mode == 'human':
            print(output)
        else:
            return output


def test_bluebook_env():
    """Test the Bluebook citation environment"""
    import os
    
    # Check if data file exists
    data_path = "/Users/kyle/Developer/cap-rlvr/data_tasks/bluebook/train.jsonl"
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Creating test environment with empty data...")
        data_path = "nonexistent.jsonl"  # Will create empty environment
    
    # Create environment with small subset for testing
    env = BluebookCitationEnv(data_path=data_path, subset_size=10)
    
    print(f"Created Bluebook environment with {env.get_sample_count()} samples")
    
    # Test reset and step
    obs = env.reset()
    print(f"Initial observation keys: {list(obs.keys())}")
    
    # Render the environment
    env.render()
    
    # Test a step with sample response
    test_response = "123 U.S. 456 (1990)"
    next_obs, reward, done, info = env.step(test_response)
    
    print(f"\nStep results:")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    env.close()


if __name__ == '__main__':
    test_bluebook_env()