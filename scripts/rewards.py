#!/usr/bin/env python3
"""
Unified reward functions module for all CAP RLVR tasks.
"""
from typing import Dict, Union, Optional
import importlib
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all reward function classes
from reward_holding import HoldingRewardFunction
from reward_bluebook import BluebookRewardFunction
from reward_irac import IRACRewardFunction
from reward_entail import EntailmentRewardFunction

try:
    from reward_retrieval import RetrievalRewardFunction
    RETRIEVAL_AVAILABLE = True
except ImportError as e:
    print(f"Retrieval reward function not available: {e}")
    RETRIEVAL_AVAILABLE = False


class UnifiedRewardFunction:
    """Unified reward function that handles all task types"""
    
    def __init__(self, faiss_index_path: Optional[str] = None):
        """
        Initialize unified reward function.
        
        Args:
            faiss_index_path: Path to FAISS index for retrieval task (optional)
        """
        # Initialize all reward functions
        self.holding_reward = HoldingRewardFunction()
        self.bluebook_reward = BluebookRewardFunction()
        self.irac_reward = IRACRewardFunction()
        self.entail_reward = EntailmentRewardFunction()
        
        # Initialize retrieval reward if available and index path provided
        self.retrieval_reward = None
        if RETRIEVAL_AVAILABLE and faiss_index_path:
            try:
                self.retrieval_reward = RetrievalRewardFunction(faiss_index_path)
            except Exception as e:
                print(f"Failed to initialize retrieval reward: {e}")
        
        # Task type detection patterns
        self.task_indicators = {
            'holding': ['choices', 'answer_idx', 'Choose the correct holding'],
            'bluebook': ['ground_truth', 'Fill in the citation', 'metadata'],
            'summarise': ['ground_truth', 'IRAC', 'Summarize this legal case'],
            'retrieval': ['positives', 'Find 5 cases', 'related_cases'],
            'entail': ['label', 'relationship', 'context', 'citing_case']
        }
    
    def detect_task_type(self, sample: Dict) -> str:
        """Automatically detect the task type from sample structure"""
        # First check if explicit task field exists (HuggingFace format)
        if 'task' in sample:
            task_value = sample['task']
            if task_value in ['entail', 'holding', 'bluebook', 'summarise', 'retrieval']:
                return task_value
        
        # Legacy detection logic for other formats
        for task_type, indicators in self.task_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in sample:
                    score += 1
                # Check in 'prompt' field for HuggingFace format
                elif isinstance(sample.get('prompt', ''), str) and indicator in sample['prompt']:
                    score += 1
                # Also check legacy 'inputs' field
                elif isinstance(sample.get('inputs', ''), str) and indicator in sample['inputs']:
                    score += 1
            
            # If majority of indicators present, classify as this task
            if score >= len(indicators) // 2 + 1:
                return task_type
        
        # Fallback: try to infer from prompt/inputs text
        prompt_text = str(sample.get('prompt', sample.get('inputs', ''))).lower()
        if 'choose the correct holding' in prompt_text:
            return 'holding'
        elif 'fill in the citation' in prompt_text:
            return 'bluebook'
        elif 'summarize' in prompt_text and 'irac' in prompt_text:
            return 'summarise'
        elif 'find' in prompt_text and 'cases' in prompt_text:
            return 'retrieval'
        elif 'relationship' in prompt_text:
            return 'entail'
        
        return 'unknown'
    
    def _extract_context_from_prompt(self, prompt: str) -> str:
        """Extract case context from prompt by removing instruction text"""
        # Look for common prompt patterns and extract the actual case context
        lines = prompt.split('\n')
        context_lines = []
        
        # Skip instruction lines and find the actual case content
        in_context = False
        for line in lines:
            line = line.strip()
            # Start capturing after common instruction patterns
            if any(marker in line.lower() for marker in ['context:', 'facts:', 'case:', 'citing case:', 'cited case:']):
                in_context = True
                # Include the line if it has content beyond the marker
                if ':' in line:
                    content_after_marker = line.split(':', 1)[1].strip()
                    if content_after_marker:
                        context_lines.append(content_after_marker)
                continue
            elif in_context and line:
                context_lines.append(line)
        
        return ' '.join(context_lines) if context_lines else prompt
    
    def _parse_holding_format(self, sample: Dict) -> Dict:
        """Parse holding task format to extract choices and answer index"""
        result = {}
        prompt = sample.get('prompt', '')
        completion = sample.get('completion', '')
        
        # Extract choices from prompt - look for A), B), C) patterns
        import re
        choice_pattern = r'([A-E])\)\s*([^A-E\n]+?)(?=[A-E]\)|$)'
        matches = re.findall(choice_pattern, prompt, re.DOTALL)
        
        if matches:
            choices = [match[1].strip() for match in matches]
            result['choices'] = choices
            
            # Find answer index from completion
            # Look for letter indicators in completion
            answer_pattern = r'\b([A-E])\b'
            answer_match = re.search(answer_pattern, completion.upper())
            if answer_match:
                answer_letter = answer_match.group(1)
                result['answer_idx'] = ord(answer_letter) - ord('A')
            else:
                # Try to match completion text with choices
                completion_lower = completion.lower().strip()
                for i, choice in enumerate(choices):
                    if completion_lower in choice.lower() or choice.lower() in completion_lower:
                        result['answer_idx'] = i
                        break
                else:
                    result['answer_idx'] = 0  # Default to first choice
        else:
            # Fallback: create dummy choices if parsing fails
            result['choices'] = [completion, 'Alternative choice']
            result['answer_idx'] = 0
        
        return result
    
    def _convert_sft_to_reward_format(self, sample: Dict, task_type: str) -> Dict:
        """
        Convert SFT dataset format to the format expected by reward functions.
        
        Args:
            sample: Sample from SFT dataset (HuggingFace format)
            task_type: Task type
            
        Returns:
            Sample in format expected by reward function
        """
        converted = sample.copy()
        
        # Handle task-specific field mappings
        if task_type == 'entail':
            # Entail reward function expects 'label' and 'context'
            converted['label'] = sample.get('expected_completion', sample.get('completion', ''))
            converted['context'] = self._extract_context_from_prompt(sample.get('inputs', sample.get('prompt', '')))
        
        elif task_type == 'bluebook':
            # Bluebook reward function expects 'ground_truth'
            converted['ground_truth'] = sample.get('expected_completion', sample.get('completion', ''))
        
        elif task_type == 'holding':
            # Holding reward function expects 'choices' and 'answer_idx'
            holding_data = self._parse_holding_format(sample)
            converted.update(holding_data)
        
        elif task_type == 'retrieval':
            # Retrieval reward function expects 'inputs' and 'positives'
            converted['inputs'] = sample.get('inputs', sample.get('prompt', ''))
            # Try to get positives from metadata, otherwise empty list
            metadata = sample.get('metadata', {})
            converted['positives'] = metadata.get('positives', metadata.get('related_cases', []))
        
        elif task_type == 'summarise':
            # IRAC reward function expects 'ground_truth' (dict or string)
            completion = sample.get('expected_completion', sample.get('completion', ''))
            # Try to structure the ground truth if it's a simple string
            if isinstance(completion, str):
                converted['ground_truth'] = {
                    'summary': completion,
                    'key_parties': [],  # Could extract from prompt if needed
                    'case_name': '',
                    'year': ''
                }
            else:
                converted['ground_truth'] = completion
        
        return converted

    def reward(self, sample: Dict, model_output: str, task_type: Optional[str] = None) -> float:
        """
        Compute reward for any task type.
        
        Args:
            sample: Task sample with inputs and ground truth
            model_output: Model's response
            task_type: Explicit task type (optional, will auto-detect if None)
            
        Returns:
            Float reward between 0.0 and 1.0
        """
        try:
            # Auto-detect task type if not provided
            if task_type is None:
                task_type = self.detect_task_type(sample)
            
            # Convert sample to format expected by reward functions
            converted_sample = self._convert_sft_to_reward_format(sample, task_type)
            
            # Route to appropriate reward function
            if task_type == 'holding':
                return self.holding_reward.reward(converted_sample, model_output)
            elif task_type == 'bluebook':
                return self.bluebook_reward.reward(converted_sample, model_output)
            elif task_type == 'summarise':
                return self.irac_reward.reward(converted_sample, model_output)
            elif task_type == 'entail':
                return self.entail_reward.reward(converted_sample, model_output)
            elif task_type == 'retrieval':
                if self.retrieval_reward is not None:
                    return self.retrieval_reward.reward(converted_sample, model_output)
                else:
                    print("Retrieval reward function not available")
                    return 0.0
            else:
                print(f"Unknown task type: {task_type}")
                return 0.0
                
        except Exception as e:
            import traceback
            print(f"ERROR in reward function for task {task_type}:")
            print(f"  Exception: {e}")
            print(f"  Sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
            print(f"  Model output length: {len(model_output) if model_output else 'None'}")
            print(f"  Full traceback:")
            traceback.print_exc()
            return 0.0
    
    def get_available_tasks(self) -> list:
        """Get list of available task types"""
        tasks = ['holding', 'bluebook', 'summarise', 'entail']
        if self.retrieval_reward is not None:
            tasks.append('retrieval')
        return tasks


def test_unified_rewards():
    """Test the unified reward function with sample data"""
    # Initialize unified reward function
    unified_reward = UnifiedRewardFunction()
    
    # Test samples for each task type
    test_samples = {
        'holding': {
            'sample': {
                'case_id': 'test_001',
                'inputs': 'Choose the correct holding from Test v. Case:',
                'choices': [
                    'The contract was valid and enforceable.',
                    'The contract was void due to impossibility.',
                    'The contract was voidable due to duress.'
                ],
                'answer_idx': 0
            },
            'response': 'A - The contract was valid and enforceable.'
        },
        
        'bluebook': {
            'sample': {
                'case_id': 'test_002',
                'inputs': 'Fill in the citation: ___ U.S. ___ (___)',
                'ground_truth': '123 U.S. 456 (1990)',
                'metadata': {'volume': '123', 'page': '456', 'year': '1990', 'reporter': 'U.S.'}
            },
            'response': '123 U.S. 456 (1990)'
        },
        
        'summarise': {
            'sample': {
                'case_id': 'test_003',
                'inputs': 'Summarize this legal case using IRAC format: [case text here]',
                'ground_truth': {
                    'summary': 'Contract dispute between parties',
                    'key_parties': ['Plaintiff', 'Defendant'],
                    'year': '1995'
                }
            },
            'response': 'Issue: Contract validity. Rule: Valid contracts require consideration. Application: Here, consideration was present. Conclusion: Contract was enforceable.'
        },
        
        'entail': {
            'sample': {
                'pair_id': 'test_004',
                'label': 'AFFIRMS',
                'context': 'The court affirmed the lower court decision, upholding the ruling.',
                'inputs': 'What is the relationship? Context: The court affirmed...'
            },
            'response': 'AFFIRMS - The court upheld the previous decision.'
        }
    }
    
    print("Testing Unified Reward Function")
    print("=" * 40)
    
    for task_type, test_data in test_samples.items():
        sample = test_data['sample']
        response = test_data['response']
        
        # Test auto-detection
        detected_task = unified_reward.detect_task_type(sample)
        reward_score = unified_reward.reward(sample, response)
        
        print(f"\nTask: {task_type}")
        print(f"Detected: {detected_task}")
        print(f"Reward: {reward_score:.3f}")
        print(f"Available tasks: {unified_reward.get_available_tasks()}")


if __name__ == '__main__':
    test_unified_rewards()