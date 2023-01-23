from tqdm import tqdm
import numpy as np
from torch import nn
import torch
import pandas as pd
np.random.seed(0)

from torch.utils.data import DataLoader

def hit_at_k(
    y_true, 
    y_pred, 
    device=None
):
    # (batch_size, )
    # (batch_size, k)
    if y_true.ndim == 1:
        y_true = torch.unsqueeze(y_true, dim=-1)
    return torch.sum(y_true == y_pred, dim=-1) # (batch_size,)

def ndcg_at_k(
    y_true,
    y_pred,
    device="cpu"
):
    if y_true.ndim == 1:
        y_true = torch.unsqueeze(y_true, dim=-1)
    where_ind = torch.where(y_true == y_pred)
    dcg_score = torch.zeros(len(y_true)).to(device) # (batch_size,)
    dcg_score[where_ind[0]] = 1. / torch.log(where_ind[1].to(torch.float) + 2)
    return dcg_score
    

def evaluate(
    test_data,
    model, 
    device,
    k=20,
    verbose=False
):
    model.eval()
    
    res = pd.DataFrame(columns=["sessionId", "Hit@%d"%k, "NDCG@%d"%k, "Reward"])
    if verbose:
         ts = tqdm(test_data, desc="[Evaluate]")   
    else:
        ts = test_data

    recommendations = []
    for batch in ts:
        sess, state, action, reward, _, _ = batch
        sess = sess.numpy()
        reward = reward.numpy()
        
        state = state.to(device)
        action = action.to(device)
        pred = model.recommend(state, k)
        hit = hit_at_k(action, pred, device).detach().to("cpu").numpy().copy()
        ndcg = ndcg_at_k(action, pred, device).detach().to("cpu").numpy().copy()
        
        recs = np.hstack([np.expand_dims(sess, -1), np.expand_dims(action.to("cpu").detach().numpy().copy(), -1), state.to("cpu").detach().numpy().copy(), pred.to("cpu").detach().numpy().copy()])
        recommendations += [recs]
        result = pd.DataFrame([sess, hit, ndcg, reward]).T
        result.columns = ["sessionId", "Hit_at_%d"%k, "NDCG_at_%d"%k, "Reward"]
        res = pd.concat([res, result], axis=0)
    
    scores = []
    reward_name = res["Reward"].unique()
    reward_name = {reward_name.max():"purchase", reward_name.min():"click"}
    for r in res["Reward"].unique():
        score = res[res["Reward"] == r][["sessionId", "Hit_at_%d"%k, "NDCG_at_%d"%k]].groupby(["sessionId"]).sum().sum() / len(res[res["Reward"] == r])
        score.index = [reward_name[r]+ "_" + ind for ind in score.index]
        scores += [score]
    scores = pd.concat(scores)
    recommendations = np.concatenate(recommendations, axis=0)     
    return res, scores, recommendations