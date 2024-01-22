import pandas as pd
import numpy as np
from fastai.metrics import mse,mae
import torch
from fastai.losses import BaseLoss
from torch.nn import CrossEntropyLoss, MSELoss
def check_type(y):
    if torch.is_tensor(y):
        if "cuda" in str(y.device):
            y=y.cpu()
        y=y.detach().numpy()
        y=pd.DataFrame(y)
    if y.index.name is None:
        y.index.name='index'
    return y
        
def weighted_mse(p,t):
    error = (t - p) ** 2
    weights = 1-t
    loss = weights * error
    return loss.mean()        


import torch

def calculate_loss_torch(y_pred, y_true, return_mean=True):
    best_predicted_score=y_pred.min(axis=1).values
    best_predicted_score_repeated=best_predicted_score.repeat(y_pred.shape[1]).view(y_pred.T.shape).T
    best_predicted_has_value_zero =y_pred.subtract(best_predicted_score_repeated)
    best_predicted_has_value_one =1-best_predicted_has_value_zero.sign()
    true_value_of_best_predicted=(best_predicted_has_value_one*y_true).max(axis=1)
    asloss=true_value_of_best_predicted.values.subtract(y_true.min(axis=1).values).sum()
    mymse=torch.sum((y_pred - y_true) ** 2)
    total_loss=asloss+mymse
    return total_loss


def calculate_misrankings_score(y_pred,y_true):
    y_true=check_type(y_true)
    y_pred=check_type(y_pred)
    total_misrankings=0
    total=0
    for a1_index, a1 in enumerate(y_true.columns):
        for a2_index, a2 in enumerate(y_true.columns[a1_index+1:]):
            if a1!=a2:
                pair_misrankings_count=((y_true[a1] < y_true[a2])!= (y_pred[a1] < y_pred[a2])).astype(int).sum()
                total+=y_true.shape[0]
                total_misrankings+=pair_misrankings_count

    return (total-total_misrankings)/total


    
def calculate_first_ranked_ranking_score(y_pred,y_true, return_mean=True):
    y_true=check_type(y_true)
    y_pred=check_type(y_pred)
    y_pred_new=y_pred.reset_index().melt(id_vars=y_true.index.names, value_vars=y_true.columns, var_name='algorithm_name').sort_values(y_true.index.names+['value']).reset_index()

    y_pred_new['algorithm_rank']=[i%len(y_true.columns) for i in y_pred_new.index]

    y_true_new=y_true.reset_index().melt(id_vars=y_true.index.names, value_vars=y_true.columns, var_name='algorithm_name').sort_values(y_true.index.names+['value']).reset_index()
    
    y_true_new=y_true_new.rename(columns={'value':'algorithm_score_true'})
    y_true_new['algorithm_rank']=[i%len(y_true.columns) for i in y_true_new.index]

    t=y_pred_new.merge(y_true_new,left_on=list(y_true.index.names)+['algorithm_name'], right_on=list(y_true.index.names)+['algorithm_name'], suffixes=['_predicted','_true'],)
    of_interest=t.query('algorithm_rank_predicted==0')
    of_interest.loc[:,'score']=[1-x for x in of_interest['algorithm_score_true'].values]
    return of_interest['score'].mean() if return_mean else of_interest

'''
def calculate_loss(y_true, y_pred,return_mean=True):
    y_true=check_type(y_true)
    y_pred=check_type(y_pred)
    index_names=list(y_true.index.names)
    y_pred_new=y_pred.reset_index().melt(id_vars=index_names, value_vars=y_true.columns, var_name='algorithm_name', value_name='algorithm_score_predicted').sort_values(index_names+['algorithm_score_predicted']).reset_index(drop=True)

    y_pred_new['algorithm_rank']=[i%len(y_true.columns) for i in y_pred_new.index]

    y_true_new=y_true.reset_index().melt(id_vars=index_names, value_vars=y_true.columns, var_name='algorithm_name', value_name='algorithm_score_true').sort_values(index_names+['algorithm_score_true']).reset_index(drop=True)

    y_true_new['algorithm_rank']=[i%len(y_true.columns) for i in y_true_new.index]

    t=y_pred_new.merge(y_true_new,left_on=index_names+['algorithm_name'], right_on=index_names+['algorithm_name'], suffixes=['_predicted','_true'],)

    predicted_best=t.query('algorithm_rank_predicted==0').copy()[list(y_true.index.names) + ['algorithm_score_true']]
    true_best=t.query('algorithm_rank_true==0').copy()[list(y_true.index.names) + ['algorithm_score_true']]

    of_interest=predicted_best.merge(true_best, left_on=index_names, right_on=index_names, suffixes=['_predicted','_true'],)
    of_interest['score']=of_interest.apply(lambda x: 1- (x['algorithm_score_true_predicted']-x['algorithm_score_true_true']), axis=1)
    return of_interest['score'].mean() if return_mean else of_interest[index_names+['score']]'''

def calculate_loss(y_pred,y_true, return_mean=True):
    y_true=check_type(y_true)
    y_pred=check_type(y_pred)
    if y_true.index.name is None or y_pred.index.name is None:
        y_true.index.name='index'
        y_pred.index.name='index'
    index_names=list(y_true.index.names)
    y_pred_new=y_pred.reset_index().melt(id_vars=index_names, value_vars=y_true.columns, var_name='algorithm_name', value_name='algorithm_score_predicted').sort_values(index_names+['algorithm_score_predicted']).reset_index(drop=True)

    y_pred_new['algorithm_rank']=[i%len(y_true.columns) for i in y_pred_new.index]

    y_true_new=y_true.reset_index().melt(id_vars=index_names, value_vars=y_true.columns, var_name='algorithm_name', value_name='algorithm_score_true').sort_values(index_names+['algorithm_score_true']).reset_index(drop=True)

    y_true_new['algorithm_rank']=[i%len(y_true.columns) for i in y_true_new.index]

    t=y_pred_new.merge(y_true_new,left_on=index_names+['algorithm_name'], right_on=index_names+['algorithm_name'], suffixes=['_predicted','_true'],)

    predicted_best=t.query('algorithm_rank_predicted==0')[list(y_true.index.names) + ['algorithm_score_true']]
    true_best=t.query('algorithm_rank_true==0')[list(y_true.index.names) + ['algorithm_score_true']]

    of_interest=predicted_best.merge(true_best, left_on=index_names, right_on=index_names, suffixes=['_predicted','_true'],)
    of_interest['score']=of_interest.apply(lambda x: 1- (x['algorithm_score_true_predicted']-x['algorithm_score_true_true']), axis=1)
    return of_interest['score'].mean() if return_mean else of_interest[index_names+['score']]
