import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
#from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import sys
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from config import *
#import autosklearn.regression
from sklearn.model_selection import train_test_split
from networkx import Graph, write_adjlist
from networkx.algorithms.dominating import dominating_set
from networkx.algorithms.mis import maximal_independent_set
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(42)



def run_selector(ela_representation_df,min_similarity_threshold=0.9,algorithm='ds',scale=True):
    ela_representation_df,_=preprocess_ela(ela_representation_df,[])
    
    ela = MinMaxScaler().fit_transform(ela_representation_df) if scale else ela_representation_df
    ela_representation_df = pd.DataFrame(ela, index=ela_representation_df.index, columns = ela_representation_df.columns)
    s = cosine_similarity(ela_representation_df.values, ela_representation_df.values)
    similarity_df = pd.DataFrame(s, index=ela_representation_df.index, columns=ela_representation_df.index)

    g = Graph()
    g.add_nodes_from(similarity_df.index)

    i = 0
    for index1, row1 in similarity_df.iterrows():
        for index2 in row1.keys():
            i += 1
            if index1 == index2:
                continue
            if row1[index2] > min_similarity_threshold:
                g.add_edge(index1, index2)
    return dominating_set(g) if algorithm=='ds' else maximal_independent_set(g)

def run_selector_on_y(ela_representation_df,min_similarity_threshold=0.9,algorithm='ds'):
 
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
            if row1[index2] > min_similarity_threshold:
                g.add_edge(index1, index2)
    return dominating_set(g) if algorithm=='ds' else maximal_independent_set(g)


def remove_correlations(df):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    df.drop(to_drop, axis=1, inplace=True)
    return df

def drop_constant_columns(dataframe):
    return dataframe.loc[:, (dataframe != dataframe.iloc[0]).any()]


def replace_invalids_with_nan(df):
    df=df.replace([np.inf, -np.inf, 'inf','-inf'], np.nan)
    #df[df.select_dtypes(include=["number","object"]).columns] = df[df.select_dtypes(include=["number","object"]).columns].replace(r'^\s*$', np.nan, regex=True)
    df = df.replace(r'^([A-Za-z]|_)+$', None, regex=True)
    return df 
    
def remove_columns_with_a_lot_of_missing_values(train, tests):
    t=train.isna().sum()/train.shape[0]
    columns_to_keep=list(t[t==0].index)
    columns_to_keep=list(set.intersection(*[set(t.columns) for  t in [train]+tests]))
    train=train[columns_to_keep]
    tests=[t[columns_to_keep] for t in tests]
    return train, tests

def preprocess_ela(train, tests, drop_constant=True):

    train=replace_invalids_with_nan(train).select_dtypes(exclude=['object'])
    tests=[replace_invalids_with_nan(t) for t in tests]

    train,tests=remove_columns_with_a_lot_of_missing_values(train, tests)
    train = train.fillna(train.median())
    
    tests = [t.fillna(train.median()) for t in tests]
    train = np.clip(train, a_max=3.4e+38 , a_min=1.2e-38) 

    tests = [ np.clip(t, a_max=3.4e+38, a_min=1.2e-38) for t in tests]
    
    if drop_constant:
        train=drop_constant_columns(train)

    tests=[t[train.columns] for t in tests]
    
  
    return train, tests

def calculate_misrankings_score(y_true, y_pred):
    total_misrankings=0
    total=0
    for a1_index, a1 in enumerate(y_true.columns):
        for a2_index, a2 in enumerate(y_true.columns[a1_index+1:]):
            if a1!=a2:
                pair_misrankings_count=((y_true[a1] < y_true[a2])!= (y_pred[a1] < y_pred[a2])).astype(int).sum()
                total+=y_true.shape[0]
                total_misrankings+=pair_misrankings_count

    return (total-total_misrankings)/total



def calculate_first_ranked_y_score(y_pred, budget, algorithm_performance,return_mean=True):
   
    
    id_columns=y_pred.index.names
    print(id_columns)
    y_pred_new=y_pred.reset_index().melt(id_vars=id_columns, value_vars=y_pred.columns, var_name='algorithm_name').sort_values(y_pred.index.names+['value']).reset_index()

    y_pred_new['algorithm_rank']=[i%len(y_pred.columns) for i in y_pred_new.index]
    algorithm_performance=algorithm_performance.query('budget==@budget').drop(columns=['seed','budget'])
    
    y_pred_new=y_pred_new.query("algorithm_rank==0")
    worst_rank=algorithm_performance['algorithm_rank'].max()
    worst_best=algorithm_performance.query('algorithm_rank==0').merge(algorithm_performance.query('algorithm_rank==@worst_rank'), left_on=id_columns,right_on=id_columns, suffixes=['_best','_worst'])

    y_pred_new=y_pred_new.rename(columns={'algorithm_name':'algorithm_name_predicted', 'algorithm_rank':'algorithm_rank_predicted'})
    m=y_pred_new.drop(columns=['index','value']).merge(worst_best.query("best_y_worst!=best_y_best"), left_on=id_columns,right_on=id_columns)

    m=m.merge(algorithm_performance, left_on=id_columns+['algorithm_name_predicted'], right_on=id_columns+['algorithm_name']).rename(columns={'best_y':'best_y_predicted', 'algorithm_rank':'true_rank_of_predicted_algorithm'})
    m['prediction_score']=m.apply(lambda x: 1-(x['best_y_predicted']-x['best_y_best'])/(x['best_y_worst']-x['best_y_best']), axis=1)
    return m['prediction_score'].mean() if return_mean else m['prediction_score']
    
    
    
    
def calculate_first_ranked_ranking_score(y_true, y_pred,return_mean=True):
    y_pred_new=y_pred.reset_index().melt(id_vars=y_true.index.names, value_vars=y_true.columns, var_name='algorithm_name').sort_values(y_true.index.names+['value']).reset_index()

    y_pred_new['algorithm_rank']=[i%len(y_true.columns) for i in y_pred_new.index]

    y_true_new=y_true.reset_index().melt(id_vars=y_true.index.names, value_vars=y_true.columns, var_name='algorithm_name').sort_values(y_true.index.names+['value']).reset_index()
    
    y_true_new=y_true_new.rename(columns={'value':'algorithm_score_true'})
    y_true_new['algorithm_rank']=[i%len(y_true.columns) for i in y_true_new.index]

    t=y_pred_new.merge(y_true_new,left_on=list(y_true.index.names)+['algorithm_name'], right_on=list(y_true.index.names)+['algorithm_name'], suffixes=['_predicted','_true'],)
    of_interest=t.query('algorithm_rank_predicted==0').copy()
    of_interest.loc[:,'score']=[1-x for x in of_interest['algorithm_score_true'].values]
    return of_interest['score'].mean() if return_mean else of_interest

def calculate_loss(y_true, y_pred,return_mean=True):
    if y_true.index.name is None or y_pred.index.name is None:
        y_true.index.name='index'
        y_pred.index.name='index'
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
    return of_interest['score'].mean() if return_mean else of_interest[index_names+['score']]

def save_predictions(y_test,y_pred,  result_file_name):
    y_test.to_csv(result_file_name+'_test.csv')
    y_pred.columns=y_test.columns
    y_pred.to_csv(result_file_name+'_pred.csv')
    
class MLModel():


    def run(self,X_train, y_train,test_names,  X_tests, y_tests,result_base_file_name, feature_name, test_precisions, budget, save_feature_importance=False):

        result_base_file_name+='_'+feature_name
        X_train, X_tests=preprocess_ela(X_train, X_tests)
        
        print("Training shape: ", X_train.shape)
        model = self.train_model(X_train, y_train)
        
        if self.name=='rf' and save_feature_importance:
            df = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
            df.sort_values(by='importance', ascending=False).to_csv(result_base_file_name + '_feature_importance.csv',index=False)


        scores=[]
        for X_test, y_test, test_name, precision in zip (X_tests, y_tests, test_names, test_precisions):
            if self.scale:
                X_test=pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)
            y_pred_test=pd.DataFrame(model.predict(X_test), index=y_test.index)
            save_predictions(y_test, y_pred_test, result_base_file_name + f'_{test_name}')
            #first_ranked_y_score=calculate_first_ranked_y_score(y_pred_test,budget,  precision, return_mean=True)
    
            
            misrankings_score=calculate_misrankings_score(y_test, y_pred_test)
            first_ranked_ranking_score=calculate_first_ranked_ranking_score(y_test, y_pred_test, return_mean=True)
            loss = calculate_loss(y_test,y_pred_test)
            scores+=[{'test_name':test_name, 'misranking_score': misrankings_score,'loss': loss,'first_ranked_score':first_ranked_ranking_score } ]
            
        
        print(feature_name)
        print(scores)
        return scores, model, X_train, X_tests
    
    def train_model(self, X_train, y_train,predict_log):
        pass
    

    
    
class RFModel(MLModel):
    name='rf'
    scale=False
    def train_model(self,X_train, y_train):
        rf_regressor =RandomForestRegressor()
        rf_regressor.fit(X_train, y_train)
        return rf_regressor


class DummyChainModel(MLModel):
    name='dummy_chain'
    scale=False
    def train_model(self,X_train, y_train):
        rf_regressor = RegressorChain(base_estimator=DummyRegressor(strategy="mean"))
        rf_regressor.fit(X_train, y_train)
        return rf_regressor       


class DummyModel(MLModel):
    name='dummy'
    scale=False
    def train_model(self,X_train, y_train):
        rf_regressor = DummyRegressor(strategy="mean")
        rf_regressor.fit(X_train, y_train)
        return rf_regressor       



 

def get_y(precision,id_columns, budget):
    return precision.query('budget==@budget').reset_index().pivot(index=id_columns,columns=['algorithm_name'],values=['algorithm_rank']).droplevel(0, axis=1)


class Benchmark:
    def __init__(self, name, id_columns, dimension,algorithm_portfolio=None, version=None, do_discrete_ranking=True,sample_count_dimension_factor=50):
        self.id_columns=id_columns
        self.name=name
        self.dimension=dimension
        self.sample_count_dimension_factor=sample_count_dimension_factor
        self.version = version if version is not None else name
        self.ela=self.initialize_ela(scale_y=False)
        self.ela_scaled_y=self.initialize_ela(scale_y=True)
        self.initialize_precision(algorithm_portfolio, do_discrete_ranking)
        self.initialize_transformer_embeddings()
        self.algorithm_portfolio=algorithm_portfolio
        self.initialize_samples()
        
        
    def initialize_transformer_embeddings(self):
        self.transformer={}
        for fold in range(0,10):
            self.transformer[fold]=pd.read_csv(f'{data_dir}/transformer_features/{self.sample_count_dimension_factor}d_samples/{self.name}_{self.dimension}d_fold_{fold}.csv', index_col=0)

    def initialize_samples(self):
        self.samples=pd.read_csv(f'{data_dir}/samples/{self.sample_count_dimension_factor}d_samples/{self.name}_{self.dimension}d_scaled.csv', index_col=0)
                
    def initialize_ela(self, scale_y=False):
        if scale_y:
            ela=pd.read_csv(f'{data_dir}/ela_features/{self.sample_count_dimension_factor}d_samples/{self.name}_{self.dimension}d_scaled.csv',index_col=[0])
        else:
            ela=pd.read_csv(f'{data_dir}/ela_features/{self.sample_count_dimension_factor}d_samples/{self.name}_{self.dimension}d.csv',index_col=[0])
        if 'f' in ela.columns:
            ela=ela.set_index('f')
        if 'Unnamed: 0' in ela.columns:
            ela=ela.drop(columns=['Unnamed: 0'])
        if scale_y:
            ela=ela.add_suffix('_sy')
        return ela

    def initialize_precision(self, algorithm_portfolio,do_discrete_ranking):
        try:
            if do_discrete_ranking:
                self.precision=pd.read_csv(f'{data_dir}/algorithm_performance/{self.version}_{self.dimension}d_{algorithm_portfolio}_ranks.csv',index_col=self.id_columns)
            else:
                self.precision=pd.read_csv(f'{data_dir}/algorithm_performance/{self.version}_{self.dimension}d_{algorithm_portfolio}_column_normalized_score.csv',index_col=self.id_columns)

        except Exception as e:
            print(e)


    def get_problems_to_use(self,budget):
        y=get_y(self.precision,self.id_columns, budget).dropna()
        transformer_ids=set.intersection(*[set(self.transformer[i].index) for i in range(0,10)])

        problems= list(set(self.ela.index).intersection(set(y.index)).intersection(set(transformer_ids)).intersection(set(self.samples.index)).intersection(set(list(self.ela.index))))
        return problems
    
    def balance_precision(self,precision):
        all_performance=pd.DataFrame()

        for a in precision['algorithm_name'].drop_duplicates().values:
            for budget in precision['budget'].drop_duplicates().values:

                fs=precision.query('algorithm_name==@a and algorithm_rank==0 and budget==@budget').sample(200, replace=False).index
                all_performance=pd.concat([all_performance,precision.query('budget==@budget').loc[fs]])
        return all_performance
    
    
    def get_all_data(self, fold, budget, balance=False):
        
        p=self.precision if not balance else self.balance_precision(self.precision)
        y=get_y(p,self.id_columns, budget)

        problems_to_use=self.get_problems_to_use(budget)
        ela=self.ela.loc[problems_to_use]
        ela_scaled_y=self.ela_scaled_y.loc[problems_to_use]
        transformer=self.transformer[fold].loc[problems_to_use]
        merged=pd.concat([transformer,ela], axis=1) #if not scale_y else pd.concat([transformer,ela_scaled_y], axis=1) 
        return {'y':y.loc[problems_to_use],
                'full_y':y, 
                'ela':ela, 
                'ela_sy':ela_scaled_y,
                'ela+ela_sy':ela.merge(ela_scaled_y,left_index=True, right_index=True),
                'transformer':transformer, 
                'merged': merged,
                'precision':self.precision.loc[problems_to_use], 
                'samples':self.samples.loc[problems_to_use]
               }
        
        return {'y':y.loc[problems_to_use],
                'full_y':y, 
                'ELA':ela,
                'ELAsy': ela_scaled_y, 
                'TransOpt':transformer, 
                'ELA+TransOpt': pd.concat([transformer,ela], axis=1), 
                'ELAsy+TransOpt': pd.concat([transformer,ela_scaled_y], axis=1), 
                'ELA+ELAsy+TransOpt': pd.concat([ela,ela_scaled_y,transformer], axis=1),
                'ELA+ELAsy': pd.concat([ela_scaled_y,ela], axis=1),
                'precision':self.precision.loc[problems_to_use], 
                'samples':self.samples.loc[problems_to_use]
               }
    
    def get_split_data(self, budget, folds=10, instance_split=False):
        if self.name=='bbob':
            problems_to_use=np.array(self.get_problems_to_use(budget))
            kf = KFold(n_splits=folds)
            train_indices=[]
            test_indices=[]
            
            
            if instance_split:
                for train_problem_instances, test_problem_instances in kf.split(list(range(1,101))):
                    train_indices+= [[f'{problem_id}_{instance_id+1}' for instance_id in train_problem_instances for problem_id in range(1,25)]]
                    test_indices+=[[f'{problem_id}_{instance_id+1}' for instance_id in test_problem_instances for problem_id in range(1,25)]]
                return list(zip(train_indices,test_indices))
            else:
                for train_problem_class, test_problem_class in kf.split(list(range(1,25))):
                    train_indices+= [[f'{problem_id+1}_{instance_id}' for problem_id in train_problem_class for instance_id in range(1,101)]]
                    test_indices+=[[f'{problem_id+1}_{instance_id}'for problem_id in test_problem_class for instance_id in range(1,101)]]
                return list(zip(train_indices,test_indices))
        
                
            
        else:
            problems_to_use=np.array(self.get_problems_to_use(budget))
            kf = KFold(n_splits=folds)
            return kf.split(problems_to_use)
    