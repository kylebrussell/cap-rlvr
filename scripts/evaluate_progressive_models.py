#!/usr/bin/env python3
"""
Automated evaluation of all progressive GRPO models.
Compares progressive training sequence performance against SFT baselines.
Uses the same evaluation methodology as logs/eval-tracking.md for consistency.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class ProgressiveModelEvaluator:
    def __init__(self):
        self.models_to_evaluate = [
            # SFT Baseline Models (for reference)
            {
                "name": "SFT 30K (Baseline)",
                "path": "models/sft_qwen3_14b_lora_30k",
                "type": "sft_baseline",
                "description": "Optimal SFT model (81.3% accuracy)"
            },
            # Progressive GRPO Models (local paths on Lambda)
            {
                "name": "Bluebook GRPO",
                "path": "models/grpo_30k_bluebook/qwen3-14b/grpo/bluebook",
                "type": "grpo_progressive",
                "base_model": "SFT",
                "description": "Citation formatting specialist (2,988 training pairs)"
            },
            {
                "name": "Holding GRPO", 
                "path": "models/grpo_30k_holding/qwen3-14b/grpo/holding",
                "type": "grpo_progressive", 
                "base_model": "Bluebook GRPO",
                "description": "Holding selection expert (3,000 training pairs)"
            },
            {
                "name": "Summarise GRPO",
                "path": "models/grpo/qwen3-14b/grpo/summarise", 
                "type": "grpo_progressive",
                "base_model": "Holding GRPO",
                "description": "IRAC summarization expert (755 training pairs)"
            },
            {
                "name": "Entail GRPO",
                "path": "models/grpo/qwen3-14b/grpo/entail",
                "type": "grpo_progressive",
                "base_model": "Summarise GRPO", 
                "description": "Case relationship classifier (840 training pairs)"
            }
        ]
        
        self.evaluation_script = "scripts/simple_eval_sft.py"
        self.tasks = ["bluebook", "holding", "summarise", "retrieval", "entail"]
        self.samples_per_task = 20  # Full evaluation (same as eval-tracking.md)
        self.pass_threshold = 0.80
        
        # Results storage
        self.results = {}
        self.output_file = f"logs/progressive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
    def check_evaluation_script(self) -> bool:
        """Check if evaluation script exists and is executable"""
        if not os.path.exists(self.evaluation_script):
            print(f"❌ Evaluation script not found: {self.evaluation_script}")
            return False
        return True
    
    def evaluate_model(self, model_info: Dict) -> Dict:
        """Evaluate a single model using the standard evaluation methodology"""
        print(f"\n🔄 Evaluating {model_info['name']}...")
        print(f"   Path: {model_info['path']}")
        print(f"   Type: {model_info['type']}")
        
        try:
            # Run evaluation script (same as used in eval-tracking.md)
            cmd = [
                "python", self.evaluation_script,
                "--model_path", model_info['path'],
                "--num_samples", str(self.samples_per_task),
                "--task", "all"
            ]
            
            print(f"   Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 60 minute timeout
            )
            
            if result.returncode != 0:
                print(f"❌ Evaluation failed for {model_info['name']}")
                print(f"   Error: {result.stderr}")
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "model_info": model_info
                }
            
            # Parse text output
            try:
                output_lines = result.stdout.strip().split('\n')
                
                # Parse task results from output lines like "✅ PASS Bluebook: 0.800 (16/20 correct)"
                task_results = {}
                total_correct = 0
                total_samples = 0
                tasks_passed = 0
                
                for line in output_lines:
                    # Look for task result lines
                    if any(task.title() in line for task in self.tasks):
                        # Extract task name, accuracy, and counts
                        for task in self.tasks:
                            if task.title() in line:
                                # Parse accuracy and counts from format like "Bluebook: 0.800 (16/20 correct)"
                                import re
                                match = re.search(rf'{task.title()}:\s*([\d\.]+)\s*\((\d+)/(\d+)', line)
                                if match:
                                    accuracy = float(match.group(1))
                                    correct = int(match.group(2))
                                    total = int(match.group(3))
                                    
                                    task_results[task] = {
                                        "accuracy": accuracy,
                                        "correct": correct,
                                        "total": total,
                                        "passed": accuracy >= self.pass_threshold
                                    }
                                    
                                    total_correct += correct
                                    total_samples += total
                                    
                                    if accuracy >= self.pass_threshold:
                                        tasks_passed += 1
                                    
                                    break
                
                # Calculate overall metrics
                overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
                
                result_summary = {
                    "status": "success", 
                    "model_info": model_info,
                    "task_results": task_results,
                    "overall_accuracy": overall_accuracy,
                    "total_correct": total_correct,
                    "total_samples": total_samples,
                    "tasks_passed": tasks_passed,
                    "total_tasks": len(task_results),
                    "stage_1_ready": tasks_passed >= 4  # Need 4/5 tasks at 80%
                }
                
                print(f"✅ {model_info['name']}: {overall_accuracy:.1%} accuracy ({total_correct}/{total_samples})")
                print(f"   Tasks passed: {tasks_passed}/{len(task_results)} (≥80%)")
                
                return result_summary
                
            except Exception as e:
                print(f"❌ Failed to parse evaluation results for {model_info['name']}")
                print(f"   Parse Error: {e}")
                print(f"   Raw output: {result.stdout[:500]}...")
                return {
                    "status": "parse_error",
                    "error": str(e),
                    "model_info": model_info
                }
                
        except subprocess.TimeoutExpired:
            print(f"❌ Evaluation timed out for {model_info['name']}")
            return {
                "status": "timeout", 
                "model_info": model_info
            }
        except Exception as e:
            print(f"❌ Unexpected error evaluating {model_info['name']}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_info": model_info
            }
    
    def run_all_evaluations(self) -> Dict:
        """Run evaluations for all progressive models"""
        print("🚀 Starting Progressive Model Evaluation")
        print(f"📊 Methodology: {self.samples_per_task} samples per task, 80% threshold")
        print(f"📝 Output file: {self.output_file}")
        print("=" * 60)
        
        if not self.check_evaluation_script():
            return {}
        
        results = {}
        
        for model_info in self.models_to_evaluate:
            model_name = model_info["name"]
            results[model_name] = self.evaluate_model(model_info)
        
        self.results = results
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive markdown report"""
        report = f"""# Progressive GRPO Model Evaluation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Methodology**: {self.samples_per_task} samples per task, {self.pass_threshold:.0%} accuracy threshold
**Evaluation Script**: `{self.evaluation_script}`

## Executive Summary

This report evaluates the progressive GRPO training sequence against SFT baselines to measure the cumulative effect of sequential improvements.

### Progressive Training Sequence
1. **SFT Base** → **Bluebook GRPO** (citation formatting mastery)
2. **Bluebook GRPO** → **Holding GRPO** (building on citation knowledge)  
3. **Holding GRPO** → **Summarise GRPO** (adding structured reasoning)
4. **Summarise GRPO** → **Entail GRPO** (completing legal reasoning suite)

## Results Summary

| Model | Type | Overall Accuracy | Tasks Passed | Stage 1 Ready | Base Model |
|-------|------|------------------|---------------|---------------|-------------|
"""
        
        # Add results table
        for model_name, result in self.results.items():
            if result.get("status") == "success":
                model_info = result["model_info"]
                overall_acc = result["overall_accuracy"]
                tasks_passed = result["tasks_passed"]
                total_tasks = result["total_tasks"]
                stage_ready = "✅ YES" if result["stage_1_ready"] else "❌ NO"
                base_model = model_info.get("base_model", "N/A")
                model_type = model_info["type"]
                
                report += f"| {model_name} | {model_type} | {overall_acc:.1%} | {tasks_passed}/{total_tasks} | {stage_ready} | {base_model} |\n"
            else:
                status = result.get("status", "unknown")
                report += f"| {model_name} | - | ❌ {status.upper()} | - | - | - |\n"
        
        report += "\n## Detailed Task Performance\n\n"
        
        # Detailed results for each model
        for model_name, result in self.results.items():
            if result.get("status") == "success":
                model_info = result["model_info"] 
                task_results = result["task_results"]
                
                report += f"### {model_name}\n\n"
                report += f"**Description**: {model_info['description']}\n"
                report += f"**Model Path**: `{model_info['path']}`\n"
                if "base_model" in model_info:
                    report += f"**Base Model**: {model_info['base_model']}\n"
                report += f"**Overall Performance**: {result['overall_accuracy']:.1%} accuracy ({result['total_correct']}/{result['total_samples']} samples)\n"
                report += f"**Stage 1 Ready**: {'✅ YES' if result['stage_1_ready'] else '❌ NO'}\n\n"
                
                report += "| Task | Accuracy | Correct/Total | Status | Notes |\n"
                report += "|------|----------|---------------|--------|---------|\n"
                
                for task in self.tasks:
                    if task in task_results:
                        task_data = task_results[task]
                        accuracy = task_data.get("accuracy", 0.0)
                        correct = task_data.get("correct", 0)
                        total = task_data.get("total", 0)
                        status = "✅ PASS" if accuracy >= self.pass_threshold else "❌ FAIL"
                        notes = task_data.get("notes", "")
                        
                        report += f"| {task.title()} | {accuracy:.3f} | {correct}/{total} | {status} | {notes} |\n"
                    else:
                        report += f"| {task.title()} | N/A | N/A | ⚠️ NO DATA | Missing evaluation data |\n"
                
                report += "\n"
            
            else:
                # Failed evaluation
                model_info = result["model_info"]
                status = result.get("status", "unknown")
                error = result.get("error", "Unknown error")
                
                report += f"### {model_name}\n\n"
                report += f"**Status**: ❌ {status.upper()}\n"
                report += f"**Error**: {error}\n"
                report += f"**Model Path**: `{model_info['path']}`\n\n"
        
        report += "## Progressive Training Analysis\n\n"
        
        # Analysis section
        successful_results = [r for r in self.results.values() if r.get("status") == "success"]
        
        if len(successful_results) >= 2:
            report += "### Performance Progression\n\n"
            
            # Extract performance trends
            grpo_models = [r for r in successful_results if r["model_info"]["type"] == "grpo_progressive"]
            
            if grpo_models:
                report += "**Progressive GRPO Training Effects:**\n\n"
                prev_accuracy = None
                
                for result in grpo_models:
                    model_name = result["model_info"]["name"]
                    accuracy = result["overall_accuracy"]
                    tasks_passed = result["tasks_passed"]
                    
                    if prev_accuracy is not None:
                        change = accuracy - prev_accuracy
                        change_str = f"({change:+.1%})"
                    else:
                        change_str = ""
                    
                    report += f"- **{model_name}**: {accuracy:.1%} accuracy, {tasks_passed} tasks passed {change_str}\n"
                    prev_accuracy = accuracy
                
                report += "\n"
        
        report += "### Key Insights\n\n"
        report += "- **Progressive Training**: Each model builds on the previous GRPO-trained model\n"
        report += "- **Task Specialization**: Models trained on specific legal reasoning tasks\n"
        report += "- **Cumulative Learning**: Sequential improvements in legal domain knowledge\n"
        report += "- **Performance Tracking**: Consistent evaluation methodology with SFT baselines\n\n"
        
        report += "## Next Steps\n\n"
        
        # Find best performing model
        best_grpo = None
        best_accuracy = 0
        
        for result in successful_results:
            if result["model_info"]["type"] == "grpo_progressive":
                if result["overall_accuracy"] > best_accuracy:
                    best_accuracy = result["overall_accuracy"] 
                    best_grpo = result
        
        if best_grpo:
            model_name = best_grpo["model_info"]["name"]
            stage_ready = best_grpo["stage_1_ready"]
            
            if stage_ready:
                report += f"1. ✅ **Proceed to Stage 1**: Use `{model_name}` for multi-task GRPO training\n"
                report += f"2. 🔄 **Multi-task Integration**: Combine all tasks for joint optimization\n"
            else:
                report += f"1. ⚠️ **Additional Training Needed**: Best model `{model_name}` not ready for Stage 1\n"
                report += f"2. 🔄 **Task-Specific Improvements**: Focus on failed tasks before multi-task training\n"
        
        report += f"3. 📊 **Update eval-tracking.md**: Add progressive model results to main tracking file\n"
        report += f"4. 🚀 **Continue Progressive Sequence**: Use evaluation insights for next training phase\n\n"
        
        report += "---\n\n"
        report += f"*Report generated by `scripts/evaluate_progressive_models.py` at {datetime.now().isoformat()}*\n"
        
        return report
    
    def save_report(self, report: str):
        """Save the evaluation report to file"""
        os.makedirs("logs", exist_ok=True)
        
        with open(self.output_file, "w") as f:
            f.write(report)
        
        print(f"📝 Report saved to: {self.output_file}")
    
    def run(self):
        """Main execution method"""
        print("🎯 Progressive GRPO Model Evaluation Suite")
        print("=" * 50)
        
        # Run all evaluations
        results = self.run_all_evaluations()
        
        if not results:
            print("❌ No evaluation results to report")
            return
        
        # Generate and save report
        print("\n📊 Generating evaluation report...")
        report = self.generate_report()
        self.save_report(report)
        
        # Print summary
        print("\n🎯 Evaluation Complete!")
        print(f"📝 Detailed report: {self.output_file}")
        
        successful_count = sum(1 for r in results.values() if r.get("status") == "success")
        total_count = len(results)
        
        print(f"📊 Results: {successful_count}/{total_count} models evaluated successfully")
        
        # Print quick summary
        print("\n📈 Quick Summary:")
        for model_name, result in results.items():
            if result.get("status") == "success":
                accuracy = result["overall_accuracy"]
                tasks_passed = result["tasks_passed"]
                total_tasks = result["total_tasks"]
                ready = "✅" if result["stage_1_ready"] else "❌"
                print(f"   {ready} {model_name}: {accuracy:.1%} ({tasks_passed}/{total_tasks} tasks)")
            else:
                status = result.get("status", "unknown")
                print(f"   ❌ {model_name}: {status}")

def main():
    evaluator = ProgressiveModelEvaluator()
    evaluator.run()

if __name__ == "__main__":
    main()