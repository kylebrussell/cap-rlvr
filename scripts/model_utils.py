#!/usr/bin/env python3
"""
Model Utilities for CAP RLVR Training

Provides utilities for extracting model information and generating 
consistent output paths with model size information.
"""

import re
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def extract_model_size(model_path: str) -> Optional[str]:
    """
    Extract model size from model path or name.
    
    Supports various naming conventions:
    - Qwen/Qwen3-14B-Instruct → 14B
    - models/qwen3-7b-sft → 7B  
    - /path/to/qwen-32b → 32B
    - custom-72B-model → 72B
    
    Args:
        model_path: Path to model directory or HuggingFace model name
        
    Returns:
        Model size string (e.g., "14B", "7B") or None if not found
    """
    # Common patterns for model sizes
    size_patterns = [
        r'(\d+\.?\d*[Bb])',  # 14B, 7B, 32B, 1.8B
        r'(\d+\.?\d*)-?[Bb]',  # 14-B, 7-B  
        r'qwen3?-(\d+\.?\d*[Bb])',  # qwen3-14B, qwen-7B
        r'(\d+\.?\d*)(?:[Bb]|billion)',  # 14B, 7billion
    ]
    
    model_path_str = str(model_path).lower()
    
    for pattern in size_patterns:
        match = re.search(pattern, model_path_str, re.IGNORECASE)
        if match:
            size = match.group(1)
            # Normalize to uppercase B
            if not size.upper().endswith('B'):
                size += 'B'
            return size.upper()
    
    logger.warning(f"Could not extract model size from: {model_path}")
    return None

def generate_output_path(base_dir: str, model_path: str, task_name: str = None, 
                        stage: str = None, suffix: str = None) -> Path:
    """
    Generate consistent output paths with model size information.
    
    Examples:
        - models/qwen3-14B/sft/bluebook
        - models/qwen3-7B/grpo/stage0_complete
        - models/qwen3-32B/grpo/production_ready
        
    Args:
        base_dir: Base output directory (e.g., "models")
        model_path: Source model path to extract size from
        task_name: Optional task name (bluebook, holding, etc.)
        stage: Optional stage name (stage0_complete, production_ready)
        suffix: Optional additional suffix
        
    Returns:
        Pathlib Path object with model size included
    """
    base_path = Path(base_dir)
    
    # Extract model size
    model_size = extract_model_size(model_path)
    if model_size:
        model_dir = f"qwen3-{model_size.lower()}"
    else:
        # Fallback to generic name if size not detected
        model_dir = "qwen3-unknown"
        logger.warning(f"Using fallback model directory: {model_dir}")
    
    # Build path components
    path_components = [model_dir]
    
    # Add training type (sft or grpo)
    if "sft" in base_dir.lower() or any(x in str(base_dir).lower() for x in ["supervised", "fine-tun"]):
        path_components.append("sft")
    else:
        path_components.append("grpo")
    
    # Add task name if provided
    if task_name:
        path_components.append(task_name)
    
    # Add stage if provided
    if stage:
        path_components.append(stage)
    
    # Add suffix if provided
    if suffix:
        path_components.append(suffix)
    
    return base_path / Path(*path_components)

def get_deployment_name(model_path: str, stage: str = "production") -> str:
    """
    Generate deployment-ready model name with size information.
    
    Args:
        model_path: Source model path
        stage: Deployment stage (production, staging, etc.)
        
    Returns:
        Deployment name (e.g., "qwen3-cap-rlvr-14B-production")
    """
    model_size = extract_model_size(model_path)
    size_suffix = f"-{model_size.lower()}" if model_size else ""
    
    return f"qwen3-cap-rlvr{size_suffix}-{stage}"

def parse_model_info(model_path: str) -> Tuple[Optional[str], str]:
    """
    Parse model path to extract size and base model name.
    
    Args:
        model_path: Path to model
        
    Returns:
        Tuple of (model_size, base_model_family)
    """
    model_size = extract_model_size(model_path)
    
    # Determine base model family
    path_lower = str(model_path).lower()
    if 'qwen3' in path_lower or 'qwen-3' in path_lower:
        base_family = 'qwen3'
    elif 'qwen' in path_lower:
        base_family = 'qwen'
    else:
        base_family = 'unknown'
    
    return model_size, base_family

# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_paths = [
        "Qwen/Qwen3-14B-Instruct",
        "models/qwen3-7b-sft", 
        "/path/to/qwen-32b",
        "custom-72B-model",
        "qwen3-1.8B-chat",
        "some-random-model"
    ]
    
    print("Model Size Extraction Tests:")
    print("-" * 40)
    for path in test_paths:
        size = extract_model_size(path)
        print(f"{path:<30} → {size or 'Not found'}")
    
    print("\nOutput Path Generation Tests:")
    print("-" * 40)
    
    test_cases = [
        ("models/grpo", "Qwen/Qwen3-14B-Instruct", "bluebook", None, None),
        ("models/grpo", "qwen3-7b-sft", None, "stage0_complete", None),
        ("models/sft", "qwen-32B", "holding", None, "checkpoint-1000"),
    ]
    
    for base_dir, model_path, task, stage, suffix in test_cases:
        output_path = generate_output_path(base_dir, model_path, task, stage, suffix)
        print(f"{model_path} → {output_path}")
    
    print("\nDeployment Name Tests:")
    print("-" * 40)
    for path in test_paths[:3]:
        deploy_name = get_deployment_name(path, "production")
        print(f"{path} → {deploy_name}")