#!/usr/bin/env python3
"""
Gym environment for holding selection task (multiple choice).
"""
from typing import Dict, Any, List, Optional
from .base_env import BaseCapRLVREnv


class HoldingSelectionEnv(BaseCapRLVREnv):
    """
    Environment for legal holding selection task.
    
    Task: Given a legal case, select the correct holding statement 
    from multiple choice options.
    """
    
    def __init__(self, 
                 data_path: str = "data_tasks/holding/train.jsonl",
                 max_episode_length: int = 1,
                 subset_size: Optional[int] = None):
        """
        Initialize holding selection environment.
        
        Args:
            data_path: Path to holding task JSONL file
            max_episode_length: Steps per episode (default: 1)
            subset_size: Limit dataset size for faster training
        """
        super().__init__(
            data_path=data_path,
            task_type='holding',
            max_episode_length=max_episode_length,
            subset_size=subset_size
        )
    
    def _get_ground_truth(self) -> Any:
        """Get ground truth answer for current sample"""
        if not self.current_sample:
            return None
        
        return {
            'answer_idx': self.current_sample.get('answer_idx', 0),
            'correct_choice': self.current_sample.get('choices', [])[
                self.current_sample.get('answer_idx', 0)
            ] if self.current_sample.get('choices') else None,
            'choices': self.current_sample.get('choices', [])
        }
    
    def _format_choices(self, choices: List[str]) -> str:
        """Format multiple choice options for display"""
        formatted = "\n"
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            formatted += f"{letter}. {choice}\n"
        return formatted
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment with formatted multiple choice prompt"""
        obs = super().reset()
        
        # Format the input to include properly formatted multiple choice
        if self.current_sample and 'choices' in self.current_sample:
            base_input = self.current_sample.get('inputs', '')
            choices = self.current_sample['choices']
            formatted_choices = self._format_choices(choices)
            
            # Combine base prompt with formatted choices
            full_prompt = f"{base_input}\n{formatted_choices}\nSelect your answer:"
            obs['inputs'] = full_prompt
            
        return obs
    
    def render(self, mode='human') -> Optional[str]:
        """Render holding selection environment"""
        if not self.current_sample:
            return "No current sample"
        
        output = f"=== Holding Selection Task ===\n"
        output += f"Case ID: {self.current_sample.get('case_id', 'unknown')}\n"
        output += f"Step: {self.current_step}/{self.max_episode_length}\n\n"
        
        # Show the question
        output += f"Question: {self.current_sample.get('inputs', 'No input')}\n\n"
        
        # Show choices
        if 'choices' in self.current_sample:
            output += "Choices:\n"
            output += self._format_choices(self.current_sample['choices'])
        
        # Show correct answer if available
        if 'answer_idx' in self.current_sample:
            correct_idx = self.current_sample['answer_idx']
            correct_letter = chr(ord('A') + correct_idx)
            output += f"\nCorrect Answer: {correct_letter}\n"
        
        if mode == 'human':
            print(output)
        else:
            return output


def test_holding_env():
    """Test the holding selection environment"""
    import os
    
    # Check if data file exists
    data_path = "/Users/kyle/Developer/cap-rlvr/data_tasks/holding/train.jsonl"
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Creating test environment with empty data...")
        data_path = "nonexistent.jsonl"  # Will create empty environment
    
    # Create environment with small subset for testing
    env = HoldingSelectionEnv(data_path=data_path, subset_size=10)
    
    print(f"Created holding environment with {env.get_sample_count()} samples")
    
    # Test reset and step
    obs = env.reset()
    print(f"Initial observation keys: {list(obs.keys())}")
    
    # Render the environment
    env.render()
    
    # Test a step with sample response
    test_response = "A"
    next_obs, reward, done, info = env.step(test_response)
    
    print(f"\nStep results:")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    env.close()


if __name__ == '__main__':
    test_holding_env()