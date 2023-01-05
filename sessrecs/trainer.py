from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
from torch import nn
import torch
np.random.seed(0)

#from sessrecs.logger import LossLogger
from sessrecs.evaluator import evaluate

from torch.utils.data import DataLoader
    

class BaseTrainer(object):
    
    def __init__(
        self, 
        model,
        device,
        logger,
        losslogger=None
    ):
        self.model= model
        self.losslogger = losslogger
        self.logger = logger
        self.device = device
        
        self.logger.info("Model Trainer Constructed.")
    
    def fit(
        self, 
        train_data:DataLoader,
        test_data:Optional[DataLoader]=None,
        epochs:Optional[int]=100,
        valid_count:Optional[int]=1,
        valid_target:Optional[str]="purchase_Hit@10",
        save_path:Optional[str]=None
    ):
        loss_hist = []
        max_valid_score = 0.
        count = 0
        result = {}
        
        self.model.train()
        for epoch in range(epochs):
            self.model.begin_epochs()
            losses = []
            with tqdm(train_data, desc="[Epoch %d]"%(epoch+1)) as ts:      
                for batch in ts:
                    loss = self.model.train_step(batch)
                    loss = loss.to('cpu').detach().numpy().copy()
                    losses += [loss]
                    result["train_loss"] = np.mean(losses)
                    ts.set_postfix(result)
                    
                if (test_data is not None) and (count%valid_count == 0):
                    res, score, _ = evaluate(test_data, self.model, self.device, k=10, verbose=False)
                    for col in score.index:
                        result[col] = score[col]
                    valid_score = score[valid_target]
                    if (max_valid_score < valid_score) & (save_path is not None):
                        self.model.save(save_path)
                        max_valid_score=valid_score
                    print(result)
                    
                    self.model.train()
                
            if self.losslogger is not None:
                self.losslogger.write_loss(result, epoch)
                
            loss_hist += [np.mean(losses)]
            self.model.end_epochs()
            count += 1
        return loss_hist
    