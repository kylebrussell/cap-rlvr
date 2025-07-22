#!/usr/bin/env python3
"""
Test script for all CAP RLVR gym environments.
"""
import sys
import os
import traceback

# Add envs directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'envs'))

from envs import (
    HoldingSelectionEnv,
    BluebookCitationEnv,
    IRACsSummaryEnv,
    CaseRetrievalEnv,
    EntailmentEnv
)


def test_environment(env_class, env_name, **kwargs):
    """Test a single environment class"""
    print(f"\n{'='*50}")
    print(f"Testing {env_name}")
    print(f"{'='*50}")
    
    try:
        # Create environment
        env = env_class(**kwargs)
        print(f"✓ Environment created successfully")
        print(f"  Dataset size: {env.get_sample_count()}")
        
        # Test reset
        obs = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation keys: {list(obs.keys())}")
        print(f"  Task type: {obs.get('task_type')}")
        
        # Test render
        print(f"\n--- Environment Render ---")
        env.render()
        
        # Test step with sample response
        sample_responses = {
            'HoldingSelectionEnv': "A",
            'BluebookCitationEnv': "123 U.S. 456 (1990)",
            'IRACsSummaryEnv': "Issue: Contract validity. Rule: Requirements. Application: Analysis. Conclusion: Valid.",
            'CaseRetrievalEnv': "Similar cases: 12345-67, 23456-78, 34567-89",
            'EntailmentEnv': "AFFIRMS - The court upheld the decision."
        }
        
        response = sample_responses.get(env_class.__name__, "Test response")
        next_obs, reward, done, info = env.step(response)
        
        print(f"\n✓ Environment step successful")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Info keys: {list(info.keys())}")
        
        # Test close
        env.close()
        print(f"✓ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing {env_name}: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False


def main():
    """Test all gym environments"""
    print("CAP RLVR Gym Environments Test Suite")
    print("=====================================")
    
    # Test configurations for each environment
    test_configs = [
        {
            'env_class': HoldingSelectionEnv,
            'env_name': 'Holding Selection Environment',
            'data_path': 'data_tasks/holding/train.jsonl',
            'subset_size': 5
        },
        {
            'env_class': BluebookCitationEnv,
            'env_name': 'Bluebook Citation Environment',
            'data_path': 'data_tasks/bluebook/train.jsonl',
            'subset_size': 5
        },
        {
            'env_class': IRACsSummaryEnv,
            'env_name': 'IRAC Summary Environment',
            'data_path': 'data_tasks/summarise/train.jsonl',
            'subset_size': 5
        },
        {
            'env_class': CaseRetrievalEnv,
            'env_name': 'Case Retrieval Environment',
            'data_path': 'data_tasks/retrieval/train.jsonl',
            'faiss_index_path': 'data_tasks/retrieval/embeddings.faiss',
            'subset_size': 5
        },
        {
            'env_class': EntailmentEnv,
            'env_name': 'Entailment Environment',
            'data_path': 'data_tasks/entail/train.jsonl',
            'subset_size': 5
        }
    ]
    
    results = []
    
    for config in test_configs:
        env_class = config.pop('env_class')
        env_name = config.pop('env_name')
        
        success = test_environment(env_class, env_name, **config)
        results.append((env_name, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("Test Summary")
    print(f"{'='*50}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for env_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{env_name:<35} {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} environments passed testing")
    
    if passed_tests == total_tests:
        print("🎉 All environments are working correctly!")
    else:
        print("⚠️  Some environments need attention.")
        
        # Note about missing data files
        print("\nNote: Failed tests may be due to missing data files.")
        print("Run the data preparation scripts first:")
        print("  cd scripts && python prep_holding_task.py")
        print("  cd scripts && python prep_bluebook_task.py")
        print("  cd scripts && python prep_summarise_task.py")
        print("  cd scripts && python prep_retrieval_task.py") 
        print("  cd scripts && python prep_entail_task.py")


if __name__ == '__main__':
    main()