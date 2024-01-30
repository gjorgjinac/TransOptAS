from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

@dataclass
class SplitData:
    x:any
    y:any
    ids:any
    name:str
    
    
class SplitterBase():
    include_tuning:bool
    fold:int
    def __init__(self,include_tuning=False,fold=None):
        self.include_tuning=include_tuning
        self.fold=fold
        
    def split_with_tune(self,sample_df):
        pass
    def split(self,sample_df):
        train,val,test,tune = self.split_with_tune(sample_df)
        if not self.include_tuning:
            return [('train',train),('val',pd.concat([val,tune])),('test',test)]
            
        return [('train',train),('val',val),('tune',test),('test',test)]
    
class SplitterBBOBLopo(SplitterBase):
    def split_with_tune(self,sample_df):
        max_instance_id=np.array([int(i.split('_')[1]) for i in sample_df.index]).max()
        test_problem=self.fold
        val_problem=((self.fold)%24)+1
        tune_probelm=((self.fold)%24)+2
        train_instance_ids, val_instance_ids, tune_instance_ids, test_instance_ids = [[f'{p}_{i}' for p in split_instance_ids for i in range(1,max_instance_id+1)] for split_instance_ids in [list(set(range(1,25)).difference({test_problem, val_problem, tune_problem})), [val_problem ],[tune_problem], [test_problem] ]]
        train_instance_ids, val_instance_ids, tune_instance_ids,test_instance_ids=[[s%24 for s in instances] for instances in [train_instance_ids, val_instance_ids, test_instance_ids]]
        train, val, tune, test =[sample_df.loc[list(set(split_instance_ids).intersection(set(sample_df.index)))] for split_instance_ids in [train_instance_ids, val_instance_ids,tune_instance_ids, test_instance_ids]]
        return train,val, tune,test
    
class SplitterBBOBProblem(SplitterBase):
    def split_with_tune(self,sample_df):
        problem_ids=list(range(1,25))
        max_instance_id=np.array([int(i.split('_')[1]) for i in sample_df.index]).max()
        train_instance_ids, test_instance_ids = train_test_split(problem_ids, test_size=6,random_state=self.fold)
        test_instance_ids, val_instance_ids = train_test_split(test_instance_ids, test_size=4,random_state=self.fold)
        val_instance_ids, tune_instance_ids = train_test_split(val_instance_ids, test_size=2,random_state=self.fold)
        
        train_instance_ids, val_instance_ids, tune_instance_ids, test_instance_ids = [[f'{p}_{i}' for p in split_instance_ids for i in range(1,max_instance_id+1)] for split_instance_ids in [train_instance_ids, val_instance_ids, tune_instance_ids, test_instance_ids] ]

        train, val, tune, test =[sample_df.loc[list(set(split_instance_ids).intersection(set(sample_df.index)))] for split_instance_ids in [train_instance_ids, val_instance_ids, tune_instance_ids, test_instance_ids]]
        return train,val, tune, test

class SplitterAffineProblem(SplitterBase):
    def affine_from_problem_classes(self,sample_df, problem_classes):
        max_instance_id=5#np.array([int(i.split('_')[2].replace('i','') ) for i in sample_df.index]).max()
        return [f'p{p1}_p{p2}_i{i}_i{i}_a{a}' for p1 in problem_classes for p2 in problem_classes for i in range(1,max_instance_id+1) for a in ([0.25,0.5,0.75] if p1!=p2 else [0])]

    def split_with_tune(self,sample_df):
        all_problems=list(set(range(1,25)))
        train_problems, test_problems=train_test_split(all_problems, test_size=6,random_state=self.fold)
        test_problems, val_problems=train_test_split(test_problems, test_size=4,random_state=self.fold)
        val_problems, tune_problems=train_test_split(val_problems, test_size=2,random_state=self.fold)
        print('Val problems')
        print(val_problems+tune_problems)
        print('Test problems')
        print(test_problems)
        train,val,tune,test=[sample_df.loc[list(set(self.affine_from_problem_classes(sample_df,pc)).intersection(sample_df.index))] for pc in [train_problems,val_problems,tune_problems,test_problems]]
        train=sample_df.drop(set(val.index).union(set(tune.index).union(set(test.index))))
        return train,val, tune,test

class SplitterBBOBInstance(SplitterBase):
    def split_with_tune(self,sample_df):
        max_instance_id=np.array([int(i.split('_')[1]) for i in sample_df.index]).max()
        instance_ids = list(range(1,max_instance_id+1))
        train_instance_ids, test_instance_ids = train_test_split(instance_ids, test_size=0.4,random_state=self.fold)
        test_instance_ids, val_instance_ids = train_test_split(test_instance_ids, test_size=0.5,random_state=self.fold)
        val_instance_ids, tune_instance_ids = train_test_split(val_instance_ids, test_size=0.5,random_state=self.fold)
        
        
        train_instance_ids, val_instance_ids, tune_instance_ids, test_instance_ids = [[f'{p}_{i}' for i in split_instance_ids for p in range(1,25)] for split_instance_ids in [train_instance_ids, val_instance_ids, tune_instance_ids, test_instance_ids] ]

        train, val, tune, test =[sample_df.loc[list(set(split_instance_ids).intersection(set(sample_df.index)))] for split_instance_ids in [train_instance_ids, val_instance_ids, tune_instance_ids, test_instance_ids]]
        return train,val, tune,test

class SplitterRandom(SplitterBase):
    def split_with_tune(self,sample_df):
        instance_ids = list(sample_df.index.drop_duplicates().values)
        train_instance_ids, test_instance_ids = train_test_split(instance_ids, test_size=0.3,random_state=self.fold)
        test_instance_ids, val_instance_ids = train_test_split(test_instance_ids, test_size=0.5,random_state=self.fold)
        val_instance_ids, tune_instance_ids = train_test_split(val_instance_ids, test_size=0.5,random_state=self.fold)
  
        train, val, tune, test =[sample_df.loc[split_instance_ids] for split_instance_ids in [train_instance_ids, val_instance_ids, tune_instance_ids, test_instance_ids]]
        print('Train/test')

        print(train.shape)
        print(test.shape)
        return train,val, tune,test