#!/usr/bin/env python3
"""
Gym environment for IRAC case summarization task.
"""
from typing import Dict, Any, Optional
from .base_env import BaseCapRLVREnv


class IRACsSummaryEnv(BaseCapRLVREnv):
    """
    Environment for legal case summarization using IRAC format.
    
    Task: Given a legal case, summarize it using the IRAC structure
    (Issue, Rule, Application, Conclusion).
    """
    
    def __init__(self, 
                 data_path: str = "data_tasks/summarise/train.jsonl",
                 max_episode_length: int = 1,
                 subset_size: Optional[int] = None):
        """
        Initialize IRAC summary environment.
        
        Args:
            data_path: Path to summarise task JSONL file
            max_episode_length: Steps per episode (default: 1)
            subset_size: Limit dataset size for faster training
        """
        super().__init__(
            data_path=data_path,
            task_type='summarise',
            max_episode_length=max_episode_length,
            subset_size=subset_size
        )
    
    def _get_ground_truth(self) -> Any:
        """Get ground truth summary for current sample"""
        if not self.current_sample:
            return None
        
        ground_truth = self.current_sample.get('ground_truth', {})
        
        return {
            'summary': ground_truth.get('summary', ''),
            'key_parties': ground_truth.get('key_parties', []),
            'year': ground_truth.get('year', ''),
            'holding': ground_truth.get('holding', ''),
            'facts': ground_truth.get('facts', ''),
            'issues': ground_truth.get('issues', [])
        }
    
    def render(self, mode='human') -> Optional[str]:
        """Render IRAC summary environment"""
        if not self.current_sample:
            return "No current sample"
        
        output = f"=== IRAC Case Summary Task ===\n"
        output += f"Case ID: {self.current_sample.get('case_id', 'unknown')}\n"
        output += f"Step: {self.current_step}/{self.max_episode_length}\n\n"
        
        # Show the summarization prompt
        output += f"Task: {self.current_sample.get('inputs', 'No input')}\n\n"
        
        # Show case information if available
        if 'case_name' in self.current_sample:
            output += f"Case Name: {self.current_sample['case_name']}\n"
        
        if 'year' in self.current_sample:
            output += f"Year: {self.current_sample['year']}\n"
        
        # Show ground truth summary if available
        if 'ground_truth' in self.current_sample:
            gt = self.current_sample['ground_truth']
            output += "\nExpected Summary Components:\n"
            
            if isinstance(gt, dict):
                if 'summary' in gt:
                    output += f"Summary: {gt['summary'][:200]}...\n"
                if 'key_parties' in gt:
                    output += f"Key Parties: {', '.join(gt['key_parties'])}\n"
                if 'holding' in gt:
                    output += f"Holding: {gt['holding'][:100]}...\n"
            else:
                output += f"Summary: {str(gt)[:200]}...\n"
        
        if mode == 'human':
            print(output)
        else:
            return output


def test_summarise_env():
    """Test the IRAC summary environment"""
    import os
    
    # Check if data file exists
    data_path = "/Users/kyle/Developer/cap-rlvr/data_tasks/summarise/train.jsonl"
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Creating test environment with empty data...")
        data_path = "nonexistent.jsonl"  # Will create empty environment
    
    # Create environment with small subset for testing
    env = IRACsSummaryEnv(data_path=data_path, subset_size=10)
    
    print(f"Created IRAC summary environment with {env.get_sample_count()} samples")
    
    # Test reset and step
    obs = env.reset()
    print(f"Initial observation keys: {list(obs.keys())}")
    
    # Render the environment
    env.render()
    
    # Test a step with sample IRAC response
    test_response = """
    Issue: Whether the contract was valid and enforceable under state law.
    
    Rule: A valid contract requires offer, acceptance, consideration, and mutual assent.
    
    Application: Here, the plaintiff made a clear offer, defendant accepted by performance, 
    consideration was present in the form of mutual promises, and both parties demonstrated 
    mutual assent through their conduct.
    
    Conclusion: The contract was valid and enforceable, and defendant's breach entitled 
    plaintiff to damages.
    """
    
    next_obs, reward, done, info = env.step(test_response)
    
    print(f"\nStep results:")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info keys: {list(info.keys())}")
    
    env.close()


if __name__ == '__main__':
    test_summarise_env()