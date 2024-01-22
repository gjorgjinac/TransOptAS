#from transformer_model import utils_runner_universal
from transformer_model import utils_runner_universal_parameter_tuning
from transformer_model.utils_performance_prediction_processor import *
from transformer_model.utils import *
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from config import *
from N_ranking_utils import *

import argparse

from N_transformer_selection_utils import *



parser = argparse.ArgumentParser(description="A script that creates a list of functions based on the input arguments")
parser.add_argument("--dimension", type=int, help="The problem dimension")

parser.add_argument("--budget", type=int, help="The budget")

parser.add_argument("--sample_count_dimension_factor", type=int, help="The number of samples to generate will be set to sample_count_dimension_factor*dimension")
parser.add_argument("--algorithms", type=str, help="Algorithms separated by a -")
parser.add_argument("--tune", type=str, help="1/0")


args = parser.parse_args()
dimension = args.dimension
sample_count_dimension_factor = args.sample_count_dimension_factor
budget = args.budget
algorithm_portfolio=args.algorithms
tune = int(args.tune)==1


x_columns=[f'x_{i}' for i in range (0,dimension)]
y_columns = ['y']

train_benchmark='random_no_constants'

benchmark=BenchmarkB(train_benchmark,id_names=['f'])
train_sample_df=benchmark.get_sample_data_with_best_algorithm(dimension, sample_count_dimension_factor, budget, algorithm_portfolio ) 
results=[]
result_dir=f'{data_dir}/results_transformer_{train_benchmark}'
os.makedirs(result_dir,exist_ok=True)
all_scores_df=pd.DataFrame()


selected=list(pd.read_csv(f"{data_dir}/samples/{sample_count_dimension_factor}d_samples/{train_benchmark}_{dimension}d_selected.csv", index_col=0)['selected'].unique())
print(len(selected))
print(len(train_sample_df.index.unique()))
print(len(set(selected).intersection(set(train_sample_df.index.unique()))))
selected=list(set(selected).intersection(set(train_sample_df.index.unique())))
train_sample_df=train_sample_df.loc[selected]


    
for fold in range(0,10):
    splitter=SplitterRandom(include_tuning=tune,fold=fold)
    data_processor=PerformancePredictionProcessor(verbose=False,splitter=splitter)
    runner=utils_runner_universal_parameter_tuning.UniversalRunner(data_processor=data_processor,
                                                  global_result_dir=f'transformer_results_{train_benchmark}_new', extra_info=f'{algorithm_portfolio}_{train_benchmark}_dim_{dimension}_samples_{sample_count_dimension_factor}_fold_{fold}_budget_{budget}', 
                                                  use_positional_encoding=False,
                                                  verbose=True, 
                                                  plot_training=False, 
                                                  d_model=20, d_k=10, d_v=10,
                                                  n_heads=3, 
                                                  n_layers=1,dropout=0.1,fc_dropout=0.1,
                                                  n_epochs=200, lr_max=0.001, id_names=train_sample_df.index.names,aggregations=['mean'],  batch_size=400)
    
    train_y=train_sample_df.drop(columns=x_columns+y_columns).reset_index().drop_duplicates()
    data = data_processor.run(train_sample_df,[],[],['f'])
    ela_rf_loss, ela_pred=train_ela_rf(benchmark,data['train'],data['test'])

    result=runner.run(train_sample_df, regenerate=True, save_embeddings=False)
    scores=result[3]
    parameters=result[4]
    scores=[(split_name, model_name, score_name, score_value) for split_scores in scores for (split_name,model_name), score_values in split_scores.items() for score_name,score_value in score_values.items()]
    scores+=[('test','ELA','Loss', ela_rf_loss)]


    
    scores_df=pd.DataFrame(scores, columns=['test_benchmark','model','score','value'])
    print(scores_df)
    scores_df['train_benchmark']=train_benchmark   
    scores_df['fold']=fold
    scores_df['sample_count_dimension_factor']=sample_count_dimension_factor
    scores_df['budget']=budget
    #scores_df['ela_loss']=ela_rf_loss
    ela_pred.to_csv(f'{result_dir}/dim_{dimension}_samples_{sample_count_dimension_factor}d_{algorithm_portfolio}_budget_{budget}_fold_{fold}_ela_predictions.csv')
    if tune:    
        for p in parameters.keys():
            scores_df[p]=parameters[p]
    all_scores_df=pd.concat([all_scores_df,scores_df])
    all_scores_df.to_csv(f'{result_dir}/dim_{dimension}_samples_{sample_count_dimension_factor}d_{algorithm_portfolio}_budget_{budget}.csv')
all_scores_df.to_csv(f'{result_dir}/dim_{dimension}_samples_{sample_count_dimension_factor}d_{algorithm_portfolio}_budget_{budget}.csv')

