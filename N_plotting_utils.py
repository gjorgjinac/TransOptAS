import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
from config import *
from N_ranking_utils import *

from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def benchmark_name_mapping(name):
    if name in ['bbob','random','affine','cec']:
        return name.upper()
    if name=="m4":
        return "ZIGZAG"
    if "random" in name:
        return "RANDOM"
    else:
        raise Exception("Benchmark name not supported")
    
def get_features(dimension=3, samples_to_take=None, affine_separately=False, feature_dir='p_ela', benchmarks = None,sample_count_dimension_factor=50,scaled=False,drop_constant=True):
    features=pd.DataFrame()
    if benchmarks is None:
        benchmarks=all_benchmarks
    for benchmark in benchmarks:
        name=f'{benchmark}_{dimension}d'
        if 'transformer' in feature_dir:
            file_location = f'{data_dir}/{feature_dir}/{sample_count_dimension_factor}d_samples/{name}_fold_0.csv' 
        else:
            file_location = f'{data_dir}/{feature_dir}/{sample_count_dimension_factor}d_samples/'+ (f'{name}_scaled.csv' if scaled else f'{name}.csv')
        f=pd.read_csv(file_location,index_col=0)
        if 'f' in f.columns:
            f=f.set_index('f')
        f['benchmark']=benchmark
        if samples_to_take is not None:
            sample_size=min(samples_to_take,f.shape[0])
            f=f.sample(sample_size)

        features=pd.concat([features, f])

    if affine_separately:
        
        file_location=f'{data_dir}/{feature_dir}/{sample_count_dimension_factor}d_samples/affine_{dimension}d.csv' if 'transformer' not in feature_dir else f'{data_dir}/{feature_dir}/affine_{dimension}d_fold_0.csv' 
        
        affine_ela=pd.read_csv(file_location,index_col=0)
        if 'f' in f.columns:
            affine_ela=affine_ela.set_index('f')
        for alpha in [0.25,0.5,0.75]:
            
            alpha_functions=list(filter(lambda x: x.endswith(str(alpha)), affine_ela.index))
            f=affine_ela.loc[alpha_functions].copy()
            f['benchmark']='affine' + "_"+str(alpha)
            sample_size=min(samples_to_take,f.shape[0])
            f=f.sample(sample_size)
            features=pd.concat([features, f])
    
    
    x=features.isna().sum(axis=1)
    x=x[x!=features.shape[1]-1]
    fs_to_use=list(x.index)
    print(len(fs_to_use))
    features=features.loc[fs_to_use]
    features_preprocessed,_=preprocess_ela(features,[],drop_constant=drop_constant)
    features_preprocessed['benchmark']=features['benchmark']
    return features,features_preprocessed

def get_common_features_from_benchmarks(dimension, benchmarks, feature_dirs, samples_to_take=100, affine_separately=False, sample_count_dimension_factor=50 ):
    all_features={}
    for feature_dir in feature_dirs:
        features=get_features(dimension,None,affine_separately, feature_dir=feature_dir, benchmarks=benchmarks,sample_count_dimension_factor=sample_count_dimension_factor)[1]
        all_features[feature_dir]=features
    
    subset_features=defaultdict(lambda: pd.DataFrame())
    problems_to_use=set.intersection(*[set(f.index) for f in all_features.values()])
    
    for feature_dir in feature_dirs:
        all_features[feature_dir]=all_features[feature_dir].loc[set(all_features[feature_dir].index).intersection(problems_to_use)]
        for benchmark in benchmarks:
            f=all_features[feature_dir].query('benchmark==@benchmark')
            sample_size=min(samples_to_take,f.shape[0])
            f=f.sample(sample_size)
            subset_features[feature_dir]=pd.concat([subset_features[feature_dir],f])
            
    return all_features, subset_features

def remove_correlated_features(df, threshold):
    # Compute the correlation matrix of the dataframe
    corr_matrix = df.corr().abs()
    # Select the upper triangle of the matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find the features that have a correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # Drop the features from the dataframe
    df.drop(to_drop, axis=1, inplace=True)
    # Return the dataframe with reduced features
    return df


def generate_2d_plot_matrix(problem, n=300):
    low,high=-5,5
    x = np.linspace(low,high,  n)
    y = np.linspace(low,high,  n)
    X, Y = np.meshgrid(x, y)
    veval = np.vstack([X.ravel(), Y.ravel()]).T

    Z = problem.evaluate(veval).reshape(n, n)
    return Z

def plot_problem(problem, n=300):
    low,high=-5,5
    x = np.linspace(low,high,  n)
    y = np.linspace(low,high,  n)
    X, Y = np.meshgrid(x, y)
    veval = np.vstack([X.ravel(), Y.ravel()]).T

    Z = problem.evaluate(veval).reshape(n, n)
    plt.figure()
    plt.imshow(Z, cmap='viridis', extent=[low,high, low,high], origin='lower')
    plt.show()