
# TransOptAS: Transformer-Based Algorithm Selection for Single-Objective Optimization


This repository contains the code for a project that aims to use transformer models to select the best algorithm for solving optimization problems. 


Usage
The training can be executed with the script transformer_performance_prediction_random.py. The code can be run from the command line with the following arguments:

•  --dimension: the problem dimension (int)

•  --budget: the budget (int)

•  --sample_count_dimension_factor: the number of samples to generate will be set to sample_count_dimension_factor*dimension (int)

•  --algorithms: the algorithms separated by a - (str)

•  --tune: 1/0 to indicate whether to tune the transformer model or not (str)

For example, to run the code with dimension 3, budget 30, sample count dimension factor 50, algorithms DE,GA,ES and PSO, and tuning enabled, the command would be:

python transformer_selection.py --dimension 3 --budget 50 --sample_count_dimension_factor 50 --algorithms DE-GA-ES-PSO --tune 1


We have to stress that in our experiments, we do not use parameter tuning since in our experiments, it had a detrimental effect on the results.
