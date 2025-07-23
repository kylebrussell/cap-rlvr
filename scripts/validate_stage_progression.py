#!/usr/bin/env python3
"""
Stage Progression Validation Script

Validates whether models meet the reward thresholds required for 
progressing to the next GRPO training stage.

Usage:
    python validate_stage_progression.py --stage 0 --model_path models/grpo/bluebook_grpo
    python validate_stage_progression.py --stage 1 --model_path models/grpo/stage0_complete --check_all_tasks
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_grpo import GRPOLegalTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StageProgressionValidator:
    """Validates model performance for stage progression"""
    
    # Stage progression thresholds
    STAGE_THRESHOLDS = {
        0: {'min_reward': 0.8, 'description': 'Individual Task Mastery'},
        1: {'min_reward': 0.75, 'description': 'Multi-Task Integration', 'max_variance': 0.2},
        2: {'min_reward': 0.85, 'description': 'Curriculum Refinement', 'max_variance': 0.15},
        3: {'min_reward': 0.90, 'description': 'Production Optimization', 'max_variance': 0.10}
    }
    
    TASK_NAMES = ['bluebook', 'holding', 'summarise', 'retrieval', 'entail']
    
    def __init__(self, faiss_index_path: str = None):
        """
        Initialize stage progression validator.
        
        Args:
            faiss_index_path: Path to FAISS index for retrieval task
        """
        self.faiss_index_path = faiss_index_path
        logger.info("Stage Progression Validator initialized")
        
    def validate_single_task(self, model_path: str, task: str, stage: int) -> Dict:
        """
        Validate a single task model against stage requirements.
        
        Args:
            model_path: Path to trained model
            task: Task name (bluebook, holding, etc.)
            stage: Current stage number
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {task} model for Stage {stage} progression")
        
        # Initialize trainer for evaluation
        trainer = GRPOLegalTrainer(
            model_path=model_path,
            task_name=task,
            faiss_index_path=self.faiss_index_path
        )
        
        # Use task-specific HuggingFace dataset for faster loading
        eval_data_path = f"kylebrussell/cap-rlvr-{task}"
        
        try:
            # Run evaluation
            eval_result = trainer.train(
                grpo_dataset_path=eval_data_path,
                eval_only=True
            )
            
            # Extract key metrics
            eval_metrics = eval_result.get('eval_results', {})
            mean_reward = eval_metrics.get('eval/rewards/mean', 0.0)
            reward_std = eval_metrics.get('eval/rewards/std', 0.0)
            
            # Check against stage thresholds
            threshold_data = self.STAGE_THRESHOLDS[stage]
            min_reward = threshold_data['min_reward']
            max_variance = threshold_data.get('max_variance', 1.0)
            
            # Validation checks
            reward_valid = mean_reward >= min_reward
            variance_valid = reward_std <= max_variance
            overall_valid = reward_valid and variance_valid
            
            result = {
                'task': task,
                'stage': stage,
                'model_path': model_path,
                'valid': overall_valid,
                'metrics': {
                    'mean_reward': mean_reward,
                    'reward_std': reward_std,
                    'reward_valid': reward_valid,
                    'variance_valid': variance_valid
                },
                'thresholds': {
                    'min_reward': min_reward,
                    'max_variance': max_variance
                },
                'recommendations': []
            }
            
            # Generate recommendations
            if not reward_valid:
                result['recommendations'].append(
                    f"Reward {mean_reward:.3f} below threshold {min_reward:.3f}. "
                    f"Continue training {task} task."
                )
            
            if not variance_valid:
                result['recommendations'].append(
                    f"Reward variance {reward_std:.3f} above threshold {max_variance:.3f}. "
                    f"Stabilize training with lower learning rate."
                )
            
            if overall_valid:
                result['recommendations'].append(
                    f"✅ {task.title()} task ready for Stage {stage + 1} progression"
                )
            
            logger.info(f"{task} validation: {'PASS' if overall_valid else 'FAIL'} "
                       f"(reward: {mean_reward:.3f}, std: {reward_std:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating {task}: {e}")
            return {
                'task': task,
                'valid': False,
                'error': str(e),
                'model_path': model_path
            }
    
    def validate_multi_task(self, model_path: str, stage: int) -> Dict:
        """
        Validate multi-task model against stage requirements.
        
        Args:
            model_path: Path to multi-task trained model
            stage: Current stage number
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating multi-task model for Stage {stage} progression")
        
        # Initialize trainer for multi-task evaluation
        trainer = GRPOLegalTrainer(
            model_path=model_path,
            task_name=None,  # Multi-task
            faiss_index_path=self.faiss_index_path
        )
        
        # Use HuggingFace dataset for unified evaluation
        eval_data_path = "kylebrussell/cap-rlvr-sft"
        
        try:
            # Run evaluation
            eval_result = trainer.train(
                grpo_dataset_path=eval_data_path,
                eval_only=True
            )
            
            # Extract metrics
            eval_metrics = eval_result.get('eval_results', {})
            overall_mean_reward = eval_metrics.get('eval/rewards/mean', 0.0)
            overall_reward_std = eval_metrics.get('eval/rewards/std', 0.0)
            
            # Task-specific metrics (if available)
            task_rewards = {}
            for task in self.TASK_NAMES:
                task_reward_key = f'eval/rewards/{task}/mean'
                if task_reward_key in eval_metrics:
                    task_rewards[task] = eval_metrics[task_reward_key]
            
            # Check against stage thresholds
            threshold_data = self.STAGE_THRESHOLDS[stage]
            min_reward = threshold_data['min_reward']
            max_variance = threshold_data.get('max_variance', 1.0)
            
            # Validation checks
            reward_valid = overall_mean_reward >= min_reward
            variance_valid = overall_reward_std <= max_variance
            
            # Check individual tasks if available
            task_balance_valid = True
            min_task_reward = float('inf')
            max_task_reward = 0.0
            
            if task_rewards:
                min_task_reward = min(task_rewards.values())
                max_task_reward = max(task_rewards.values())
                task_balance_valid = (max_task_reward - min_task_reward) <= 0.15
            
            overall_valid = reward_valid and variance_valid and task_balance_valid
            
            result = {
                'model_type': 'multi_task',
                'stage': stage,
                'model_path': model_path,
                'valid': overall_valid,
                'metrics': {
                    'overall_mean_reward': overall_mean_reward,
                    'overall_reward_std': overall_reward_std,
                    'task_rewards': task_rewards,
                    'min_task_reward': min_task_reward,
                    'max_task_reward': max_task_reward,
                    'reward_valid': reward_valid,
                    'variance_valid': variance_valid,
                    'task_balance_valid': task_balance_valid
                },
                'thresholds': {
                    'min_reward': min_reward,
                    'max_variance': max_variance,
                    'max_task_imbalance': 0.15
                },
                'recommendations': []
            }
            
            # Generate recommendations
            if not reward_valid:
                result['recommendations'].append(
                    f"Overall reward {overall_mean_reward:.3f} below threshold {min_reward:.3f}. "
                    f"Continue multi-task training."
                )
            
            if not variance_valid:
                result['recommendations'].append(
                    f"Reward variance {overall_reward_std:.3f} above threshold {max_variance:.3f}. "
                    f"Reduce learning rate or increase training stability."
                )
            
            if not task_balance_valid and task_rewards:
                worst_task = min(task_rewards, key=task_rewards.get)
                best_task = max(task_rewards, key=task_rewards.get)
                result['recommendations'].append(
                    f"Task imbalance detected: {worst_task} ({task_rewards[worst_task]:.3f}) "
                    f"vs {best_task} ({task_rewards[best_task]:.3f}). "
                    f"Apply task-specific weighting or additional training."
                )
            
            if overall_valid:
                result['recommendations'].append(
                    f"✅ Multi-task model ready for Stage {stage + 1} progression"
                )
            
            logger.info(f"Multi-task validation: {'PASS' if overall_valid else 'FAIL'} "
                       f"(reward: {overall_mean_reward:.3f}, std: {overall_reward_std:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating multi-task model: {e}")
            return {
                'model_type': 'multi_task',
                'valid': False,
                'error': str(e),
                'model_path': model_path
            }
    
    def generate_report(self, results: List[Dict], stage: int) -> str:
        """Generate a comprehensive validation report"""
        
        report_lines = [
            f"GRPO Stage {stage} Progression Validation Report",
            "=" * 50,
            f"Stage: {stage} - {self.STAGE_THRESHOLDS[stage]['description']}",
            f"Threshold: {self.STAGE_THRESHOLDS[stage]['min_reward']:.1%} reward",
            ""
        ]
        
        valid_count = sum(1 for r in results if r.get('valid', False))
        total_count = len(results)
        
        report_lines.extend([
            f"Overall Status: {valid_count}/{total_count} models passed",
            f"Progression Ready: {'✅ YES' if valid_count == total_count else '❌ NO'}",
            ""
        ])
        
        # Individual results
        for result in results:
            if 'error' in result:
                report_lines.extend([
                    f"❌ {result.get('task', 'Unknown')}: ERROR",
                    f"   Error: {result['error']}",
                    ""
                ])
                continue
            
            task_name = result.get('task', result.get('model_type', 'Unknown'))
            status = "✅ PASS" if result['valid'] else "❌ FAIL"
            metrics = result.get('metrics', {})
            
            report_lines.extend([
                f"{status} {task_name.title()}:",
                f"   Mean Reward: {metrics.get('mean_reward', metrics.get('overall_mean_reward', 0)):.3f}",
                f"   Std Dev: {metrics.get('reward_std', metrics.get('overall_reward_std', 0)):.3f}",
                f"   Model: {Path(result['model_path']).name}"
            ])
            
            # Add recommendations
            for rec in result.get('recommendations', []):
                report_lines.append(f"   → {rec}")
            
            report_lines.append("")
        
        # Next steps
        if valid_count == total_count:
            next_stage = stage + 1
            if next_stage <= 3:
                report_lines.extend([
                    f"🎉 Ready to proceed to Stage {next_stage}:",
                    f"   {self.STAGE_THRESHOLDS[next_stage]['description']}",
                    f"   Target: {self.STAGE_THRESHOLDS[next_stage]['min_reward']:.1%} reward"
                ])
            else:
                report_lines.append("🏁 All stages complete - ready for deployment!")
        else:
            report_lines.extend([
                "📋 Next Steps:",
                "   1. Address failing validations above",
                "   2. Continue training with recommended adjustments",
                "   3. Re-run validation when ready"
            ])
        
        return "\n".join(report_lines)

def main():
    parser = argparse.ArgumentParser(description='Validate GRPO stage progression')
    
    parser.add_argument('--stage', type=int, required=True, choices=[0, 1, 2, 3],
                       help='Current stage to validate (0-3)')
    parser.add_argument('--model_path', required=True,
                       help='Path to model to validate')
    parser.add_argument('--task', choices=['bluebook', 'holding', 'summarise', 'retrieval', 'entail'],
                       help='Specific task to validate (for Stage 0)')
    parser.add_argument('--check_all_tasks', action='store_true',
                       help='Check all individual tasks (for Stage 0)')
    parser.add_argument('--faiss_index', default=None,
                       help='Path to FAISS index for retrieval task')
    parser.add_argument('--output_report', default=None,
                       help='Save validation report to file')
    
    args = parser.parse_args()
    
    # Auto-detect FAISS index
    if args.faiss_index is None:
        potential_faiss = Path('data_tasks/retrieval/embeddings.faiss')
        if potential_faiss.exists():
            args.faiss_index = str(potential_faiss)
            logger.info(f"Auto-detected FAISS index: {args.faiss_index}")
    
    # Initialize validator
    validator = StageProgressionValidator(faiss_index_path=args.faiss_index)
    
    results = []
    
    if args.stage == 0:
        # Stage 0: Individual task validation
        if args.check_all_tasks:
            # Validate all tasks
            base_path = Path(args.model_path).parent
            for task in validator.TASK_NAMES:
                task_model_path = base_path / f"{task}_grpo"
                if task_model_path.exists():
                    result = validator.validate_single_task(str(task_model_path), task, args.stage)
                    results.append(result)
                else:
                    logger.warning(f"Model not found for task {task}: {task_model_path}")
        elif args.task:
            # Validate specific task
            result = validator.validate_single_task(args.model_path, args.task, args.stage)
            results.append(result)
        else:
            parser.error("For Stage 0, specify --task or use --check_all_tasks")
    
    else:
        # Stage 1+: Multi-task validation
        result = validator.validate_multi_task(args.model_path, args.stage)
        results.append(result)
    
    # Generate and display report
    report = validator.generate_report(results, args.stage)
    print(report)
    
    # Save report if requested
    if args.output_report:
        with open(args.output_report, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {args.output_report}")
    
    # Exit with appropriate code
    all_valid = all(r.get('valid', False) for r in results)
    sys.exit(0 if all_valid else 1)

if __name__ == '__main__':
    main()