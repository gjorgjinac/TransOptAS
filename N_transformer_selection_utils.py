
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from config import *
from N_ranking_utils import *

import argparse




def run_selector_euclidean(ela_representation_df,min_similarity_threshold=0.9,algorithm='ds',scale=True):
    ela_representation_df,_=preprocess_ela(ela_representation_df,[])
    
    ela = MinMaxScaler().fit_transform(ela_representation_df) if scale else ela_representation_df
    ela_representation_df = pd.DataFrame(ela, index=ela_representation_df.index, columns = ela_representation_df.columns)
    s = euclidean_distances(ela_representation_df.values, ela_representation_df.values)
    similarity_df = pd.DataFrame(s, index=ela_representation_df.index, columns=ela_representation_df.index)

    g = Graph()
    g.add_nodes_from(similarity_df.index)
    i = 0
    for index1, row1 in similarity_df.iterrows():
        for index2 in row1.keys():
            i += 1
            if index1 == index2:
                continue
            if row1[index2] < min_similarity_threshold:
                g.add_edge(index1, index2)
    return dominating_set(g) if algorithm=='ds' else maximal_independent_set(g)


class BenchmarkB():
    def __init__(self,name,id_names):
        self.name=name
        self.id_names=id_names
        
    def read_sample_df(self,dimension, sample_count_dimension_factor):
        file_to_read = f"{data_dir}/samples/{sample_count_dimension_factor}d_samples/{self.name}_{dimension}d.csv"
        return pd.read_csv(file_to_read, index_col=0)
    def initialize_ela(self,  sample_count_dimension_factor, dimension, scale_y=False):
        if scale_y:
            ela=pd.read_csv(f'{data_dir}/ela_features/{sample_count_dimension_factor}d_samples/{self.name}_{dimension}d_scaled.csv',index_col=[0])
        else:
            ela=pd.read_csv(f'{data_dir}/ela_features/{sample_count_dimension_factor}d_samples/{self.name}_{dimension}d.csv',index_col=[0])
        if 'f' in ela.columns:
            ela=ela.set_index('f')
        if 'Unnamed: 0' in ela.columns:
            ela=ela.drop(columns=['Unnamed: 0'])
        return ela.replace([np.inf,-np.inf],np.nan).dropna(axis=1)
    
    def scale_y(self,sample_df):
        new_sample_df=pd.DataFrame()

        for function in tqdm(sample_df.reset_index()['f'].drop_duplicates()):
            try:
                instance_df=sample_df.loc[function].copy()
                min_max_scaler = MinMaxScaler()
                new_sample_df=pd.concat([new_sample_df,pd.DataFrame(min_max_scaler.fit_transform(instance_df), index=instance_df.index, columns=instance_df.columns)])
            except Exception as e:
                print('failed minmax', function)
                continue
        #new_sample_df=new_sample_df.reset_index(drop=True).set_index(['f'])
        return new_sample_df

    def read_scaled_sample_df(self, dimension, sample_count_dimension_factor):
        
        scaled_file=f"{data_dir}/samples/{sample_count_dimension_factor}d_samples/{self.name}_{dimension}d_scaled.csv"
        if not os.path.isfile(scaled_file):
            print('Recalculating scaled sample')
            all_sample_df=self.read_sample_df(dimension, sample_count_dimension_factor)
            all_sample_df=self.scale_y(all_sample_df)
            all_sample_df.to_csv(scaled_file)
        else:
            print('Reading scaled sample from file')
            all_sample_df=pd.read_csv(scaled_file, index_col=[0])
        return all_sample_df

    
    def get_sample_data_with_best_algorithm(self,dimension, sample_count_dimension_factor, budget, algorithm_portfolio ):
        sample_df=self.read_scaled_sample_df(dimension, sample_count_dimension_factor)
        self.ela=self.initialize_ela(sample_count_dimension_factor, dimension,scale_y=False)
    
        if self.name=="autoopt":
            algorithm_performance=pd.read_csv(f'{data_dir}/algorithm_performance/{self.name}_{dimension}d_{algorithm_portfolio}_column_normalized_score.csv', index_col=0)
            algorithm_performance['algorithm_name']=algorithm_performance.apply(lambda row: f"{row['algorithm']+row['config']}", axis=1)
        
            self.algorithm_performance=algorithm_performance.query('budget==@budget').query('budget==@budget').reset_index().pivot(index='f',columns='algorithm_name',values='algorithm_rank')
        
            self.algorithm_performance.index=[c.replace('/I001/D010','') for c in self.algorithm_performance.index]
            
        else:
            algorithm_performance=pd.read_csv(f'{data_dir}/algorithm_performance/{self.name}_{dimension}d_{algorithm_portfolio}_column_normalized_score.csv', index_col=0).rename(columns={'config':'algorithm_name'})
        
            self.algorithm_performance=algorithm_performance.query('budget==@budget').query('budget==@budget').reset_index().pivot(index='f',columns='algorithm_name',values='algorithm_rank')
        
        sample_df=sample_df.merge(self.algorithm_performance,left_index=True, right_index=True)

        sample_df=sample_df.dropna()
        count=dimension* sample_count_dimension_factor
        to_keep=sample_df.reset_index().groupby('f').count().query('x_0==@count').index

        return sample_df.loc[to_keep]
    
    def get_sample_data_with_best_algorithm_and_ela(self,dimension, sample_count_dimension_factor, budget, algorithm_portfolio ):
        sample_df=self.get_sample_data_with_best_algorithm(dimension, sample_count_dimension_factor, budget, algorithm_portfolio )
        sample_df=sample_df.merge(self.ela, left_index=True, right_index=True)
        return sample_df
    
    
def train_ela_rf(benchmark,train_split,test_split):
    train_X,test_X=benchmark.ela.loc[train_split.ids].dropna(), benchmark.ela.loc[test_split.ids]
    train_X,test_X=preprocess_ela(train_X,[test_X])
    test_X=test_X[0]
    train_y,test_y=benchmark.algorithm_performance.loc[train_X.index], benchmark.algorithm_performance.loc[test_X.index]
    model=RandomForestRegressor()
    model.fit(train_X,train_y)
    pred=pd.DataFrame(model.predict(test_X), columns=test_y.columns, index=test_y.index)
    print('RF:')
    loss=calculate_loss(test_y,pred)
    print(loss)
    return loss, pred

