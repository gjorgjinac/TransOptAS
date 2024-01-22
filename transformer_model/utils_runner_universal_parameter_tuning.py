from .tsai_custom import *
from tsai.all import *
computer_setup()
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss, MSELoss
from fastai.losses import BaseLoss
import sklearn 
from fastai.callback.tracker import EarlyStoppingCallback,SaveModelCallback
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from .utils import *
from .utils_ranking_metrics import *
from .model_stats import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import os
from .config import *
from fastai.metrics import mse
import colorcet as cc
import matplotlib.pyplot as plt
from optuna.samplers import *
import matplotlib
matplotlib.cm.register_cmap("my_cmap", my_cmap)

import optuna
from optuna.integration import FastAIV1PruningCallback

from fastai import vision
    
    
class UniversalRunner():

    logger = None
    verbose: bool
    plot_training: bool
    n_heads:int 
    n_layers:int 
    d_model:int
    d_k:int
    d_v:int 
    n_epochs:int
    bs:int

    model:any
    learner:any
    result_dir:str
    use_positional_encoding:bool
    
    def __init__(self,data_processor,extra_info=None, verbose=True, lr_max=0.0005, plot_training=True, use_positional_encoding=False, n_heads=1, n_layers=1, d_model=20, d_k=10, d_v=10, n_epochs=100, batch_size=8, fold=None, iteration_count=None, include_iteration_in_x=False, train_seeds=None, val_seeds=None, global_result_dir='results', aggregations=None, id_names=None, split_type=None, dropout=0.1, fc_dropout=0.1):

        self.verbose = verbose
        self.lr_max=lr_max
        self.plot_training=plot_training
        self.n_heads=n_heads
        self.n_layers=n_layers
        self.d_model=d_model
        self.dropout=dropout
        self.fc_dropout=fc_dropout
        self.d_k=d_k
        self.d_v=d_v
        self.id_names=id_names
        self.n_epochs=n_epochs
        self.batch_size=batch_size
        self.train_seeds=train_seeds
        self.val_seeds=val_seeds

        self.data_processor=data_processor
        self.extra_info=extra_info
        self.extra_info += f'_n_heads_{n_heads}_n_layers_{n_layers}_d_model_{d_model}_d_k_{d_k}_d_v_{d_v}_aggregations_{"all" if aggregations is None else "-".join(aggregations)}'

        self.result_dir = os.path.join(global_result_dir,self.data_processor.task_name,self.extra_info)
        self.fold=fold
        self.use_positional_encoding=use_positional_encoding
        self.iteration_count=iteration_count
        self.aggregations=aggregations
        os.makedirs(self.result_dir, exist_ok = True) 
        



    def find_learning_rate(self, dls):
        print('Determining learning rate')
        learn = Learner(dls, self.model, loss_func=LabelSmoothingCrossEntropyFlat(), metrics=[RocAuc(), accuracy],  cbs=ShowGraphCallback2())
        learn.lr_find()


    def hyperparemeter_tuning_trial(self,trial,dls, tune_data):
        callbacks=[EarlyStoppingCallback( min_delta=0.001,patience=5), SaveModelCallback (monitor='valid_loss', comp=None, min_delta=0.001,
                    fname=self.extra_info, every_epoch=False, at_end=False, with_opt=False, reset_on_fit=True)]
        
        model=OptTransStats(dls.vars, dls.c, dls.len, do_regression=self.data_processor.task_name=='performance_prediction', d_model=trial.suggest_int("d_model", 20, 100,step=10), 
                                                  d_k=trial.suggest_int("d_k", 10, 50,step=10),
                                                  d_v=trial.suggest_int("d_v", 10, 50,step=10),
                                                  fc_size=trial.suggest_int("fc_size", 20, 100,step=20),
                                                  n_heads=trial.suggest_int("n_heads", 1, 3), 
                                                  d_ff=trial.suggest_int("d_ff", 20, 100,step=20),
                                                  n_layers=trial.suggest_int("n_layers", 1, 3), n_epochs=200, normalize=False, reduce=False
                                                 #[trial.suggest_categorical("aggregations",["mean","max","min","std"])]
                           )
        learner = Learner(dls, model, loss_func=calculate_loss_torch, metrics=[mse], cbs=callbacks)
        learner.fit(self.n_epochs, lr=self.lr_max) #trial.suggest_float("lr_max", 0.00001, 0.01, step=0.0005
        probas, targets, preds = learner.get_X_preds(np.swapaxes(tune_data.x,1,2), with_decoded=True, bs=self.batch_size)
        transformer_scores = self.data_processor.evaluate_model(tune_data.y,probas,tune_data.ids,None)
        return transformer_scores['Loss']
        
    def hyperparameter_tuning(self,dls, tune_data):
        prune=True
        pruner = optuna.pruners.MedianPruner() if prune else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler())
        study.optimize(partial(self.hyperparemeter_tuning_trial,dls=dls,tune_data=tune_data), n_trials=20, timeout=1200, n_jobs=1)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        return trial.params
    
    
    def train_model(self, dls, tune_data=None):
        print('Training model')
        callbacks=[EarlyStoppingCallback( min_delta=0.001,patience=5), SaveModelCallback (monitor='valid_loss', comp=None, min_delta=0.001,
                    fname=self.extra_info, every_epoch=False, at_end=False, with_opt=False, reset_on_fit=True)]
        #callbacks=[]
        lr=self.lr_max
        parameters=[{"n_heads":self.n_heads, "n_layers":self.n_layers, "d_model":self.d_model, "d_k":self.d_k, "d_v":self.d_v}]
        if tune_data is not None:
            
            parameters =self.hyperparameter_tuning(dls, tune_data)
            print('Training with tuned parameters:')
            print(parameters)
            #lr=parameters['lr_max']
            self.model=OptTransStats(dls.vars, dls.c, dls.len, n_heads=parameters['n_heads'], n_layers=parameters['n_layers'], d_model=parameters['d_model'], d_k=parameters['d_k'], d_v=parameters['d_v'], use_positional_encoding=self.use_positional_encoding, iteration_count=self.iteration_count, aggregations=["mean","max","min","std"], fc_size=parameters['fc_size'], d_ff=parameters['d_ff'] , do_regression=self.data_processor.task_name=='performance_prediction')
        else:
            print('Training with externally specified parameters')
            self.model=OptTransStats(dls.vars, dls.c, dls.len, n_heads=self.n_heads, n_layers=self.n_layers, d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, use_positional_encoding=self.use_positional_encoding, iteration_count=self.iteration_count, aggregations=self.aggregations, dropout=self.dropout, fc_dropout=self.fc_dropout, do_regression=self.data_processor.task_name=='performance_prediction')
        print(self.model.parameters() )
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total parameters with gradient")
        print(total_params)
        
        if self.plot_training:
            callbacks+=[ShowGraphCallback2()]
        if self.data_processor.task_name!='performance_prediction':
            self.learner = Learner(dls, self.model, loss_func=CrossEntropyLoss(), metrics=[ accuracy],  cbs=callbacks)
        else:
            #self.learner = Learner(dls, self.model, loss_func=calculate_loss_torch, metrics=[weighted_mse, mse, calculate_misrankings_score, calculate_loss], cbs=callbacks) 
            self.learner = Learner(dls, self.model, loss_func=calculate_loss_torch, metrics=[mse,calculate_loss_torch], cbs=callbacks) 

        start = time.time()
        self.learner.fit(self.n_epochs, lr=lr)
         
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(self.model.parameters())
        print("Total parameters with gradient")
        print(total_params)
        
        print('\nElapsed time:', time.time() - start)
        if self.plot_training:
            self.learner.plot_metrics()
        return parameters

    def evaluate(self, split_data:SplitData ):
        print('Evaluating model')
        probas, targets, preds = self.learner.get_X_preds(np.swapaxes(split_data.x,1,2), with_decoded=True, bs=self.batch_size)


        if split_data.name=='train':
            print(split_data.y.mean())
            print('Training dummy')
            self.data_processor.train_dummy(pd.DataFrame(split_data.y,index=split_data.ids))

        dummy_scores = self.data_processor.evaluate_dummy(split_data.y,split_data.ids, os.path.join(self.result_dir,f'{split_data.name}_dummy'))
        transformer_scores = self.data_processor.evaluate_model( split_data.y,probas,split_data.ids, os.path.join(self.result_dir,f'{split_data.name}_transopt'))

        return {(split_data.name,'transformer'):transformer_scores, (split_data.name, 'dummy'):dummy_scores}



    def run (self,sample_df, plot_embeddings=False, save_embeddings=True, regenerate=False):
            
        data = self.data_processor.run(sample_df, self.train_seeds, self.val_seeds, self.id_names)
        print(data.keys())
        dsets = {split_name: TSDatasets(np.swapaxes(split_data.x,1,2),split_data.y) for split_name, split_data in data.items()}
        
        dls = TSDataLoaders.from_dsets(dsets['train'], dsets['val'], bs=self.batch_size)
        
        test_data_loader = TSDataLoaders.from_dsets(dsets['test'], bs=self.batch_size)[0]
        
        dls.c=len(set(data['train'].y)) if self.data_processor.task_name!='performance_prediction' else data['train'].y[0].shape[0]
        
        print(f'Number of samples: {dls.len}, Number of variables: {dls.vars}, Number of classes: {dls.c}')
 
        parameters = self.train_model(dls, data['tune'] if 'tune' in data.keys() else None)
        torch.save(self.model, os.path.join(self.result_dir,f'trained_model.pt'))
        torch.save(self.model.state_dict(), os.path.join(self.result_dir,f'trained_model_dict.pt'))
        
        return self.model, self.learner, None,[self.evaluate(split_data) for split_data in data.values()], parameters
    
    def evaluate_other_test(self,sample_df,name):
        data= self.data_processor.process_test_only(sample_df,name)
        return self.evaluate(data )
    
    

        