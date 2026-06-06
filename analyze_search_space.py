"""
Script to analyze the AutoML search space size.
Calculates the number of possible configurations for each model.
"""

import sys
sys.path.insert(0, 'AutoML')

from automl.configspace_search_space import get_classification_configspace
from automl.components import get_classification_components
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    NormalFloatHyperparameter,
)


def calculate_discrete_configurations(param):
    """
    Calculate discrete configurations for a hyperparameter.
    For continuous parameters, we estimate based on log scale or bounds.
    """
    if isinstance(param, CategoricalHyperparameter):
        # Categorical: number of choices
        return len(param.choices)
    
    elif isinstance(param, UniformIntegerHyperparameter):
        # Integer: count of integers in range
        return param.upper - param.lower + 1
    
    elif isinstance(param, (UniformFloatHyperparameter, NormalFloatHyperparameter)):
        # Continuous: estimate using reasonable discretization
        # For log-scale, use logarithmic discretization
        # For linear scale, use 100-1000 discrete values as estimate
        if param.log:
            # Log scale: estimate ~100 meaningful steps
            return 100
        else:
            # Linear scale: estimate ~1000 meaningful steps
            return 1000
    
    return 1


def analyze_configspace():
    """Analyze the full configspace and print statistics."""
    print("=" * 80)
    print("AutoML SEARCH SPACE ANALYSIS")
    print("=" * 80)
    print()
    
    # Get all components
    components = get_classification_components()
    models = sorted(components.keys())
    
    print(f"Total Models: {len(models)}")
    print(f"Models: {', '.join(models)}")
    print()
    print("=" * 80)
    print("DETAILED BREAKDOWN BY MODEL")
    print("=" * 80)
    print()
    
    total_configs = 0
    
    for model_name in models:
        component = components[model_name]
        hyperparams = component.build_hyperparameters()
        
        # Calculate configurations for this model
        model_configs = 1
        param_details = []
        
        for param in hyperparams:
            param_name = param.name.split(":")[-1]  # Remove model_name prefix
            num_values = calculate_discrete_configurations(param)
            model_configs *= num_values
            
            # Collect parameter details
            if isinstance(param, CategoricalHyperparameter):
                param_type = f"Categorical ({len(param.choices)} choices)"
            elif isinstance(param, UniformIntegerHyperparameter):
                param_type = f"Integer [{param.lower}, {param.upper}]"
            elif isinstance(param, UniformFloatHyperparameter):
                log_str = "(log scale)" if param.log else "(linear scale)"
                param_type = f"Float [{param.lower:.2e}, {param.upper:.2e}] {log_str}"
            else:
                param_type = param.__class__.__name__
            
            param_details.append({
                'name': param_name,
                'type': param_type,
                'values': num_values
            })
        
        total_configs += model_configs
        
        print(f"Model: {model_name}")
        print(f"  Number of hyperparameters: {len(hyperparams)}")
        
        if hyperparams:
            print(f"  Hyperparameters:")
            for detail in param_details:
                print(f"    - {detail['name']:<25} {detail['type']:<40} ({detail['values']:>10,} values)")
        else:
            print(f"  Hyperparameters: None (fixed model)")
        
        print(f"  Configurations: {model_configs:,}")
        print()
    
    print("=" * 80)
    print("SEARCH SPACE SUMMARY")
    print("=" * 80)
    print()
    print(f"Total combinations (sum of all models): {total_configs:,}")
    print()
    
    # The search space is very large
    cs = get_classification_configspace()
    print(f"Total hyperparameters in config space: {len(cs.get_hyperparameters())}")
    print()
    
    # Print some interesting statistics
    avg_configs_per_model = total_configs / len(models)
    print(f"Average configurations per model: {avg_configs_per_model:,.0f}")
    
    # Get models with most and least configs
    model_configs_list = []
    for model_name in models:
        component = components[model_name]
        hyperparams = component.build_hyperparameters()
        model_configs = 1
        for param in hyperparams:
            model_configs *= calculate_discrete_configurations(param)
        model_configs_list.append((model_name, model_configs))
    
    model_configs_list.sort(key=lambda x: x[1])
    
    print()
    print("Models with SMALLEST search space:")
    for name, configs in model_configs_list[:5]:
        print(f"  {name:<25} {configs:>15,} configurations")
    
    print()
    print("Models with LARGEST search space:")
    for name, configs in model_configs_list[-5:]:
        print(f"  {name:<25} {configs:>15,} configurations")
    
    print()
    print("=" * 80)
    print("NOTES")
    print("=" * 80)
    print("""
1. Continuous hyperparameters are estimated:
   - Log-scale parameters: ~100 meaningful discrete values
   - Linear-scale parameters: ~1000 meaningful discrete values
   
2. The actual search space is MUCH larger due to:
   - Continuous parameter ranges allow infinite configurations
   - Conditional hyperparameters add complexity
   - Cross-validation adds stochasticity
   
3. To explore this huge space efficiently:
   - Bayesian optimization with surrogate models (Random Forest)
   - Acquisition functions (Expected Improvement)
   - Meta-learning warm starts
   - Parallel batch evaluation
   
4. In 50 trials with 3 parallel workers:
   - ~16-17 batches of evaluations
   - Each model with same hyperparameters NEVER runs twice
   - Smart acquisition function guides to high-performing regions
""")
    

if __name__ == "__main__":
    analyze_configspace()
