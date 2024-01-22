
import sklearn 
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.multioutput import MultiOutputRegressor
from sklearn.dummy import DummyRegressor
from .utils import *
from .utils_ranking_metrics import *

class DummyNoChainModel():

    def train_model(self,X_train, y_train):
        rf_regressor = MultiOutputRegressor(DummyRegressor(strategy="mean"))
        rf_regressor.fit(X_train, y_train)
        return rf_regressor    

class PerformancePredictionProcessor():
    verbose:bool

    algorithm_name_dict: dict

    include_iteration_in_x:bool
    task_name='performance_prediction'
    splitter:SplitterBase
    
    def __init__(self, splitter:SplitterBase, verbose=False):
  
        self.verbose=verbose
        self.algorithm_name_dict = None
 
        self.splitter=splitter

        
    
    def get_x_y(self, sample_df, split_name):
        if self.verbose:
            print('Extracting x and y')
        xys = []
        x_columns = [col for col in sample_df.columns if col.startswith('x_')]
        self.x_y_columns = x_columns + ['y']
        self.performance_columns = list(set(sample_df.columns).difference(set(self.x_y_columns)))

        for problem_id, problem_samples in tqdm(sample_df.groupby(level=0)):
            if problem_samples.shape[0]==0:
                print('No samples ', problem_id)
                continue
            x=problem_samples[self.x_y_columns].values
            xys.append((x, problem_samples[self.performance_columns].values[0], problem_id))
            
        x=np.array([xy[0] for xy in xys])
        y=np.array([xy[1] for xy in xys])

        ids=np.array([xy[2] for xy in xys])
        y=torch.as_tensor(y).type(torch.float32)
        
        return SplitData(x=x,y=y,ids=ids,name=split_name)


    
    def run(self,sample_df,train_seed, val_seed,id_names):
        split_data = self.splitter.split(sample_df)
        data = {split_name: self.get_x_y(split_data, split_name) for split_name, split_data in split_data}

        
        return data
    
    
    def process_test_only(self,sample_df,name):
        data = self.get_x_y(sample_df, name)
        return data
    
    def evaluate_model(self, y,probas,ids,save_to_file=None):
        error=mean_squared_error(y,probas)
        probas=pd.DataFrame(probas, columns=self.performance_columns)
        probas.index.name='index'
        y=pd.DataFrame(y, columns= self.performance_columns)
        y.index.name='index'
        misrankings_score=calculate_misrankings_score(probas,y)
        loss_score=calculate_loss(probas,y)
        
        
        y.columns=[f'true_{c}' for c in self.performance_columns]
        probas.columns=[f'pred_{c}' for c in self.performance_columns]
        d=pd.concat([y,probas],axis=1)
        d.index=ids

        if save_to_file is not None:
            d.to_csv(save_to_file + 'ys_predictions.csv')

        return {'MSE': error, 'Misrankings score': misrankings_score, 'Loss': loss_score}
    
    def train_dummy(self, y_train):
        self.dummy_model=DummyNoChainModel().train_model(y_train.index,y_train)
    
    def evaluate_dummy(self, y_test, test_ids,save_to_file=None):
        y_pred=self.dummy_model.predict(test_ids)
        return self.evaluate_model(y_test, y_pred,test_ids,save_to_file)
  
