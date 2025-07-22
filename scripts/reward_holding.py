#!/usr/bin/env python3
"""
Reward function for holding selection task (multiple choice).
"""
import re
from typing import Dict, List, Optional
from difflib import SequenceMatcher


class HoldingRewardFunction:
    def __init__(self):
        """Initialize holding selection reward function"""
        pass
    
    def extract_choice_from_response(self, response: str, num_choices: int) -> Optional[int]:
        """Extract the selected choice from model response"""
        response = response.strip()
        
        # Pattern 1: Look for explicit choice indicators (A, B, C, D, etc.)
        choice_patterns = [
            r'\b([A-E])\b',  # Single letter choices
            r'choice\s+([A-E])',  # "choice A"
            r'option\s+([A-E])',  # "option B"
            r'answer\s+([A-E])',  # "answer C"
            r'\(([A-E])\)',  # "(A)"
        ]
        
        for pattern in choice_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                choice_letter = matches[-1].upper()  # Take the last match
                choice_idx = ord(choice_letter) - ord('A')
                if 0 <= choice_idx < num_choices:
                    return choice_idx
        
        # Pattern 2: Look for numeric choices (1, 2, 3, etc.)
        numeric_patterns = [
            r'\b(\d+)\b',  # Any number
            r'choice\s+(\d+)',  # "choice 1"
            r'option\s+(\d+)',  # "option 2"
            r'answer\s+(\d+)',  # "answer 3"
        ]
        
        for pattern in numeric_patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    choice_num = int(matches[-1])  # Take the last match
                    choice_idx = choice_num - 1  # Convert to 0-based index
                    if 0 <= choice_idx < num_choices:
                        return choice_idx
                except ValueError:
                    continue
        
        # Pattern 3: Text similarity matching
        # If no explicit choice found, try to match against choice text
        return None
    
    def compute_text_similarity(self, response: str, choices: List[str]) -> int:
        """Find the choice with highest text similarity to response"""
        max_similarity = 0.0
        best_choice = 0
        
        response_clean = response.lower().strip()
        
        for i, choice in enumerate(choices):
            choice_clean = choice.lower().strip()
            
            # Compute similarity using difflib
            similarity = SequenceMatcher(None, response_clean, choice_clean).ratio()
            
            # Also check for substring matches
            if choice_clean in response_clean or response_clean in choice_clean:
                similarity += 0.3  # Bonus for substring match
            
            # Check for key phrase matches
            choice_words = set(choice_clean.split())
            response_words = set(response_clean.split())
            word_overlap = len(choice_words.intersection(response_words))
            if len(choice_words) > 0:
                word_similarity = word_overlap / len(choice_words)
                similarity += word_similarity * 0.2
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_choice = i
        
        return best_choice if max_similarity > 0.3 else 0
    
    def reward(self, sample: Dict, model_output: str) -> float:
        """
        Main reward function for holding selection task.
        
        Args:
            sample: Dictionary with 'choices' and 'answer_idx'
            model_output: Model's response indicating choice selection
            
        Returns:
            Float reward: 1.0 for correct choice, 0.0 for incorrect
        """
        try:
            choices = sample['choices']
            correct_answer_idx = sample['answer_idx']
            num_choices = len(choices)
            
            if num_choices == 0:
                return 0.0
            
            # Try to extract explicit choice from response
            predicted_choice = self.extract_choice_from_response(model_output, num_choices)
            
            # If no explicit choice found, use text similarity
            if predicted_choice is None:
                predicted_choice = self.compute_text_similarity(model_output, choices)
            
            # Binary reward: 1.0 for correct, 0.0 for incorrect
            if predicted_choice == correct_answer_idx:
                return 1.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error computing holding reward: {e}")
            return 0.0


def test_holding_reward():
    """Test the holding selection reward function"""
    reward_fn = HoldingRewardFunction()
    
    # Test case with explicit choice
    sample1 = {
        'choices': [
            'The contract was valid and enforceable.',
            'The contract was void due to impossibility.',
            'The contract was voidable due to duress.',
            'The contract was terminated by mutual consent.'
        ],
        'answer_idx': 0
    }
    
    # Test different response formats
    responses = [
        "The answer is A",
        "Choice A: The contract was valid and enforceable",
        "Option 1",
        "I choose (A)",
        "The contract was valid and enforceable based on the facts presented.",
        "B - The contract was void"
    ]
    
    for i, response in enumerate(responses):
        reward = reward_fn.reward(sample1, response)
        expected = 1.0 if i in [0, 1, 2, 3, 4] else 0.0  # Last one is wrong choice
        print(f"Response {i+1}: {reward:.1f} (expected {expected:.1f})")


if __name__ == '__main__':
    test_holding_reward()