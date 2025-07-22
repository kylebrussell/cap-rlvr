#!/usr/bin/env python3
"""
Base gym environment for CAP RLVR tasks.
"""
import gym
import json
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import sys
import os

# Add scripts directory to path for reward functions
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from rewards import UnifiedRewardFunction


class BaseCapRLVREnv(gym.Env, ABC):
    """
    Base environment for CAP RLVR legal reasoning tasks.
    
    All specific task environments inherit from this base class.
    """
    
    def __init__(self, 
                 data_path: str,
                 task_type: str,
                 max_episode_length: int = 1,
                 faiss_index_path: Optional[str] = None,
                 subset_size: Optional[int] = None):
        """
        Initialize base CAP RLVR environment.
        
        Args:
            data_path: Path to task data (JSONL file)
            task_type: Type of task ('holding', 'bluebook', 'summarise', 'retrieval', 'entail')
            max_episode_length: Maximum steps per episode (usually 1 for legal tasks)
            faiss_index_path: Path to FAISS index for retrieval task
            subset_size: Limit dataset to N samples for faster training/testing
        """
        super().__init__()
        
        self.data_path = data_path
        self.task_type = task_type
        self.max_episode_length = max_episode_length
        self.current_step = 0
        self.current_sample = None
        
        # Load dataset
        self.dataset = self._load_dataset(subset_size)
        print(f"Loaded {len(self.dataset)} samples for {task_type} task")
        
        # Initialize reward function
        self.reward_fn = UnifiedRewardFunction(faiss_index_path)
        
        # Define observation and action spaces (text-based)
        self.observation_space = gym.spaces.Dict({
            'inputs': gym.spaces.Text(max_length=10000),
            'task_type': gym.spaces.Discrete(5),  # 5 different task types
            'sample_id': gym.spaces.Text(max_length=100)
        })
        
        # Action space is text generation (represented as text)
        self.action_space = gym.spaces.Text(max_length=2000)
        
    def _load_dataset(self, subset_size: Optional[int] = None) -> List[Dict]:
        """Load dataset from JSONL file"""
        dataset = []
        
        if not os.path.exists(self.data_path):
            print(f"Warning: Data file not found at {self.data_path}")
            return dataset
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        dataset.append(sample)
                        
                        if subset_size and len(dataset) >= subset_size:
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON line: {e}")
                        continue
        
        return dataset
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation"""
        self.current_step = 0
        
        if not self.dataset:
            # Empty dataset fallback
            self.current_sample = {
                'inputs': f"No data available for {self.task_type} task",
                'sample_id': 'empty'
            }
        else:
            # Sample a random instance from dataset
            self.current_sample = random.choice(self.dataset)
        
        return self._get_observation()
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Text response from the model
            
        Returns:
            observation, reward, done, info
        """
        self.current_step += 1
        
        # Compute reward using unified reward function
        if self.current_sample:
            reward = self.reward_fn.reward(self.current_sample, action, self.task_type)
        else:
            reward = 0.0
        
        # Episode is done after one step for most legal tasks
        done = self.current_step >= self.max_episode_length
        
        # Additional info for debugging/analysis
        info = {
            'task_type': self.task_type,
            'sample_id': self.current_sample.get('sample_id', 'unknown') if self.current_sample else 'none',
            'step': self.current_step,
            'model_response': action,
            'ground_truth': self._get_ground_truth()
        }
        
        observation = self._get_observation() if not done else {}
        
        return observation, reward, done, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        if not self.current_sample:
            return {
                'inputs': "No sample available",
                'task_type': self._get_task_type_id(),
                'sample_id': 'none'
            }
        
        return {
            'inputs': self.current_sample.get('inputs', ''),
            'task_type': self._get_task_type_id(),
            'sample_id': str(self.current_sample.get('case_id', self.current_sample.get('sample_id', 'unknown')))
        }
    
    def _get_task_type_id(self) -> int:
        """Convert task type to numeric ID"""
        task_mapping = {
            'holding': 0,
            'bluebook': 1,
            'summarise': 2,
            'retrieval': 3,
            'entail': 4
        }
        return task_mapping.get(self.task_type, 0)
    
    @abstractmethod
    def _get_ground_truth(self) -> Any:
        """Get ground truth for current sample (task-specific)"""
        pass
    
    def render(self, mode='human') -> Optional[str]:
        """Render environment state"""
        if not self.current_sample:
            return "No current sample"
        
        output = f"Task: {self.task_type}\n"
        output += f"Step: {self.current_step}\n"
        output += f"Sample ID: {self.current_sample.get('case_id', 'unknown')}\n"
        output += f"Input: {self.current_sample.get('inputs', '')[:200]}...\n"
        
        if mode == 'human':
            print(output)
        else:
            return output
    
    def close(self):
        """Clean up environment resources"""
        pass
    
    def get_sample_count(self) -> int:
        """Get total number of samples in dataset"""
        return len(self.dataset)
    
    def get_current_sample(self) -> Optional[Dict]:
        """Get current sample for inspection"""
        return self.current_sample