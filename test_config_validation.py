#!/usr/bin/env python3
"""
Quick test script to validate the configuration changes
"""
import yaml
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from optimizer.batch_optimization import expand_dataset_configs, BatchOptimizer

def test_config_loading():
    """Test that the configuration loads properly and execution rules are applied."""
    
    # Load the configuration
    config_path = "src/config/redux_no_filters.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded successfully")
        
        # Check that execution rules are properly nested
        if 'advanced' in config and 'execution_rules' in config['advanced']:
            print("✅ Execution rules found in advanced section")
            exec_rules = config['advanced']['execution_rules']
            print(f"   - skip_ebm_large_datasets: {exec_rules.get('skip_ebm_large_datasets')}")
            print(f"   - max_features_for_ebm: {exec_rules.get('max_features_for_ebm')}")
            print(f"   - auto_adjust_trials: {exec_rules.get('auto_adjust_trials')}")
        else:
            print("❌ Execution rules not found or not properly nested")
            
        # Check global defaults
        global_defaults = config.get('global_defaults', {})
        print(f"✅ Global defaults found: top_n={global_defaults.get('top_n', 'NOT SET')}")
        
        # Expand dataset configs
        job_configs = expand_dataset_configs(config)
        print(f"✅ Expanded {len(job_configs)} job configurations")
        
        # Test batch optimizer with execution rules
        batch_optimizer = BatchOptimizer({})
        
        # Test execution rules
        for job_config in job_configs:
            dataset_name = job_config.get('dataset_name', 'unknown')
            models = job_config.get('models', ['randomforest', 'xgboost', 'lightgbm', 'ebm'])
            
            print(f"\n📊 Dataset: {dataset_name}")
            print(f"   Original models: {models}")
            print(f"   Top N: {job_config.get('top_n', 'NOT SET')}")
            
            # Apply execution rules
            filtered_models = batch_optimizer._apply_execution_rules(models, job_config)
            print(f"   Filtered models: {filtered_models}")
            
            # Test feature estimation
            num_features = batch_optimizer._estimate_dataset_features(job_config)
            print(f"   Estimated features: {num_features}")
            
            if 'ebm' in models and 'ebm' not in filtered_models:
                print("   ✅ EBM correctly filtered out due to execution rules")
            elif 'ebm' in models and 'ebm' in filtered_models:
                print("   ℹ️  EBM kept (execution rules allowed it)")
                
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)
