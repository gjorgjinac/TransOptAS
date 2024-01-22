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


class SplitterRandom(SplitterBase):
    def split_with_tune(self,sample_df):
        instance_ids = list(sample_df.index.unique())
        train_instance_ids, test_instance_ids = train_test_split(instance_ids, test_size=0.30,random_state=self.fold)
        test_instance_ids, val_instance_ids = train_test_split(test_instance_ids, test_size=0.5,random_state=self.fold)
        val_instance_ids, tune_instance_ids = train_test_split(val_instance_ids, test_size=0.5,random_state=self.fold)
  
        train, val, tune, test =[sample_df.loc[split_instance_ids] for split_instance_ids in [train_instance_ids, val_instance_ids, tune_instance_ids, test_instance_ids]]
        print('Train/test')

        print(train.shape)
        print(test.shape)
        return train,val, tune,test