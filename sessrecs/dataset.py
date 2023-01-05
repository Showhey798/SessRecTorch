from typing import Dict, Optional, Any
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pickle
np.random.seed(0)

from torch.utils.data import Dataset
import gc

from tqdm import tqdm

tqdm.pandas()

HOME = os.environ["HOME"] + "/work"


def split_data(data:Dict[str, Any], train_rate:Optional[float]=0.8, shuffle:Optional[bool]=True):
    """
    mdpデータセットとして作成したdataを分割する

    Args:
        data (Dict[str, Any]): mdp_dataset
        train_rate (Optional[float], optional): _description_. Defaults to 0.8.
        shuffle (Optional[bool], optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    key = list(data.keys())[0]
    ind = np.arange(data[key].shape[0])
    if shuffle:
        np.random.shuffle(ind)
    split_ind = int(len(ind)*train_rate)
    train_ind, test_ind = ind[:split_ind], ind[split_ind:]
    train_data, test_data = {}, {}
    for key in data.keys():
        train_data[key] = data[key][train_ind]
        test_data[key] = data[key][test_ind]
    return train_data, test_data
    

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def padding_item(a, window, pad_item=0):
    b = np.repeat(pad_item, repeats=(window-1))
    a = np.hstack([b, a])
    return a

def get_done(x):
    x = np.zeros_like(x)
    x[-1]=1
    return x

class MDPDataset(Dataset):
    
    def __init__(
        self, 
        data:Dict[str, np.ndarray],
    ):
        super().__init__()
        self.data = data
        self.len = data[list(data.keys())[0]].shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return tuple([self.data[key][index] for key in self.data.keys()])
    
def preprocess_data(datasetname="YahooR3", test_size=0.1, valid_size=0.1):
    file_path = Path("%s/dataset/%s/"%(HOME, datasetname))
    if datasetname == "YahooR3":
        dtrain = pd.read_csv(file_path/"ydata-ymusic-rating-study-v1_0-train.txt", sep="\t", header=None)
        dtest = pd.read_csv(file_path/"ydata-ymusic-rating-study-v1_0-test.txt", sep="\t", header=None)
        dtrain.columns = ["userId", "itemId", "rating"]
        dtest.columns = ["userId", "itemId", "rating"]
        dtrain["userId"] -= 1
        dtrain["itemId"] -= 1
        dtest["userId"] -= 1
        dtest["itemId"] -= 1
        
        dvalid, dtest = train_test_split(dtest, test_size=valid_size)
        
        num_users, num_items = dtrain["userId"].unique().shape[0], dtrain["itemId"].unique().shape[0]
        
        user_item_count = dtrain.groupby("userId")["itemId"].count() # ユーザーごとに観測したアイテム数
        user_item_count = user_item_count.reset_index()
        max_user = user_item_count["itemId"].max()
        user_item_count["pscores"] = user_item_count["itemId"] / max_user
        user_item_count.drop(["itemId"], axis=1, inplace=True)
        dtrain = pd.merge(dtrain, user_item_count, on="userId", how="left")
    
    elif datasetname == "ml-100k":
        dtrain = pd.read_csv(file_path/"rating.csv")
        dtrain["userId"] -= 1
        dtrain["itemId"] -= 1
        dtrain, dtest = train_test_split(dtrain, test_size=valid_size)
        
        dvalid, dtest = train_test_split(dtest, test_size=valid_size)
        num_users, num_items = dtrain["userId"].unique().shape[0], dtrain["itemId"].unique().shape[0]
        
        user_item_count = dtrain.groupby("userId")["itemId"].count() # ユーザーごとに観測したアイテム数
        user_item_count = user_item_count.reset_index()
        max_user = user_item_count["itemId"].max()
        user_item_count["pscores"] = user_item_count["itemId"] / max_user
        user_item_count.drop(["itemId"], axis=1, inplace=True)
        dtrain = pd.merge(dtrain, user_item_count, on="userId", how="left")
        
    elif datasetname == "ml-1m":
        dtrain = pd.read_csv(file_path/"ratings.csv", header=None)
        dtrain.columns = ["userId", "itemId", "rating", "timestamp"]
        dtrain["userId"] -= 1
        dtrain["itemId"] -= 1
        dtrain, dtest = train_test_split(dtrain, test_size=valid_size)
        
        dvalid, dtest = train_test_split(dtest, test_size=valid_size)
        num_users, num_items = dtrain["userId"].unique().shape[0], dtrain["itemId"].unique().shape[0]
        
        user_item_count = dtrain.groupby("userId")["itemId"].count() # ユーザーごとに観測したアイテム数
        user_item_count = user_item_count.reset_index()
        max_user = user_item_count["itemId"].max()
        user_item_count["pscores"] = user_item_count["itemId"] / max_user
        user_item_count.drop(["itemId"], axis=1, inplace=True)
        dtrain = pd.merge(dtrain, user_item_count, on="userId", how="left")

    return dtrain, dvalid, dtest, num_users, num_items


def delete_sess_and_items(
    df,
    sess_len:int, 
    item_len:int
):
    """データフレームの中からセッションの長さがsess_len未満のセッションとアイテムの数がitem_len未満のものを削除する

    Args:
        df (pd.core.dataframe.DataFrame): [sessionId, itemId, timestamp]の列を持つデータフレーム
        sess_len (int): 削除するセッションの長さの閾値
        item_len (int): 削除するアイテムの長さの閾値
    """

    before_len = len(df)
    current_len = 0
    while before_len != current_len:
        sessions = df.groupby("sessionId")["itemId"].nunique() # 各セッションの長さ
        sessions = sessions[sessions > sess_len].index
        df = df[df["sessionId"].isin(sessions)]
        items = df.groupby("itemId")["sessionId"].nunique()
        items = items[items > item_len].index
        df = df[df["itemId"].isin(items)]
        before_len = current_len
        current_len = len(df)
        
    return df

def session_preprocess_data(
    datasetname="RC15", 
    seq_len=10,
    logger=None,
    num_sessions=200000,
    rp=1.,
    rc=0.2,
    preprocess:Optional[bool]=False,
):
    file_path = Path("%s/dataset/%s/"%(HOME, datasetname))
    if datasetname == "RC15":
        if (not os.path.exists(file_path / "derived/preprocess_data.df")) or (preprocess):
            # 前処理が終わっていない場合
            if logger:
                logger.info("Preprocessing data...")
            train_path = file_path / "yoochoose-clicks.dat"
            train_buy = file_path / "yoochoose-buys.dat"
            #test_path = file_path / "yoochoose-test.dat"
            train_df = pd.read_csv(train_path, header=None)
            buy_df = pd.read_csv(train_buy, header=None)
            train_df.columns = ["sessionId", "timestamp", "itemId", "categoryId"]
            buy_df.columns = ["sessionId", "timestamp", "itemId", "price", "quantity"]
            buy_df.drop(["price", "quantity"], axis=1, inplace=True)
            train_df.drop("categoryId", axis=1, inplace=True)
            train_df["reward"] = rc
            buy_df["reward"] = rp
          
            train_df = delete_sess_and_items(train_df, 3, 1) # 長さが3未満のセッションと5回未満の出現回数のアイテムを削除
            
            buy_df = buy_df[buy_df["sessionId"].isin(train_df["sessionId"].unique())]
            buy_sessions = buy_df["sessionId"].unique()
            sessions = train_df["sessionId"].unique()

            if len(buy_sessions) < num_sessions:
                # buyのデータセットは全て使用
                sampling_sessions = num_sessions - len(buy_sessions)
                random_sessions = np.random.choice(sessions, size=sampling_sessions) # num_sessionsをサンプリング
                random_sessions = np.concatenate([random_sessions, buy_sessions])
            else:
                random_sessions = np.random.choice(buy_sessions, size=num_sessions)
            
            train_df = train_df[train_df["sessionId"].isin(random_sessions)]
            buy_df = buy_df[buy_df["sessionId"].isin(random_sessions)] # clickデータに存在するセッションのみを抽出
            df = pd.concat([train_df, buy_df], axis=0)
            if logger:
                logger.info("Delete session and items")
                logger.info("Loaded Data")
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S.%fZ")

            item_encoder = LabelEncoder()
            session_encoder = LabelEncoder()
            df = df.sort_values("timestamp")
            
            df["sessionId"] = session_encoder.fit_transform(df["sessionId"])
            df["itemId"] = item_encoder.fit_transform(df["itemId"])
            
            if logger :
                logger.info("preprocess end.")
            
            df.to_pickle(file_path / "derived/preprocess_data.df")
        else:
            df = pd.read_pickle(file_path / "derived/preprocess_data.df")

        num_items = df["itemId"].unique().max() + 1
        num_clicks = len(df[df["reward"] == rc])
        num_purchases = len(df[df["reward"] == rp])
        
        
        if logger:
            logger.info("%s Data Description Number of sessions:%d, Number of items: %d, Number of clicks : %d, Number of purchases : %d"%(datasetname, num_sessions, num_items, num_clicks, num_purchases))
        
    
    if (os.path.exists("%s/dataset/%s/derived/train_valid.df"%(HOME, datasetname))) and (not preprocess):
        if logger:
            logger.info("Loading Train Data")
        del df
        gc.collect()
        with open("%s/dataset/%s/derived/train_valid.df"%(HOME, datasetname), "rb") as f:
            train_data =  pickle.load(f)
        return train_data, num_items
    
    sess_df = pd.DataFrame(columns=["sessionId", "itemId", "reward"])
    for name, group in tqdm(df.groupby("sessionId")[["itemId", "reward"]], desc="creating sessions"):
        del_dup_items = group[~group[["itemId", "reward"]].duplicated()]
        tmp = pd.Series(
            {
                "sessionId": name, 
                "itemId" : group["itemId"].values, 
                "reward" : group["reward"].values
            })
        sess_df = pd.concat([sess_df, pd.DataFrame(tmp).T], axis=0)
    df = sess_df.reset_index().copy()
    
    df["itemId"] = df["itemId"].progress_apply(
        lambda x: rolling_window(padding_item(x, seq_len, 0), seq_len))    
    df["sessionId"] = df[["sessionId", "itemId"]].progress_apply(
        lambda x: np.repeat(x[0], repeats=x[1].shape[0]-1), axis=1)
    df["action"] = df["itemId"].progress_apply(lambda x: x[1:, -1])
    df = df[df["sessionId"].apply(lambda x: len(x)>1)]
    df["done"] = df["sessionId"].apply(get_done)
    df["reward"] = df["reward"].progress_apply(lambda x: x[1:])
    
    
    state = np.vstack(df["itemId"].apply(lambda x: x[:-1, :]).tolist())
    n_state = np.vstack(df["itemId"].apply(lambda x: x[1:, :]).tolist())
    action = np.concatenate(df["action"].tolist(), axis=0)
    done = np.concatenate(df["done"].tolist(), axis=0)
    reward = np.concatenate(df["reward"].tolist(), axis=0)
    sess = np.concatenate(df["sessionId"].tolist(), axis=0)
    
    train_data = {
        "session":sess, 
        "state": state, 
        "action": action, 
        "reward" : reward,
        "n_state": n_state,
        "done": done
    }

    with open("%s/dataset/RC15/derived/train_valid.df"%HOME, "wb") as files:
        pickle.dump(train_data, files)

    
    if logger is not None:
        logger.info("train data preprocessd.")
    
    return train_data, num_items
    

if __name__ == "__main__":
    import sys
    sys.path.append("%s/tf1_models"%HOME)
    from sessrecs.logger import create_logger
    logger = create_logger("session_data_preprocess")
    train, num_items = session_preprocess_data(seq_len=10, logger=logger)
    print(num_items)
    print("Preprocessed Data")