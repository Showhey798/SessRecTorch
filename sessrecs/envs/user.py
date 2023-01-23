from typing import Optional
import numpy as np

class User():
    """
        シミュレーション環境におけるユーザの振る舞い
        ユーザは推薦リストから0 or 1つアイテムを選択する
    """
    
    def __init__(
        self,
        embed_dim:int, # ユーザの興味座標の次元数
        alpha:float, # 興味の更新感度
        sigma:Optional[float]=0.01, # 興味の範囲
        mean:Optional[float]=0.,
        scale:Optional[float]=0.1
    ):
        self.feature_embedding = np.random.normal(loc=mean, scale=scale, size=embed_dim)
        
        self.alpha = alpha
        self.sigma = sigma
    
    def _update_state(
        self, 
        item_emb:int # (embed_dim)
    ):
         diff = item_emb - self.feature_embedding
         self.feature_embedding += self.alpha*diff
        
    
    def response(
        self, 
        item_embs:np.ndarray # (candidate_num, embed_dim)
    ):
        assert item_embs.shape[1] == self.feature_embedding.shape[0]
        
        dist = np.sum(np.square(item_embs - self.feature_embedding.reshape(1, -1)), axis=-1) # (candidate_num,)
        
        # 興味の範囲内にあるか
        isin_interest = (dist <= self.sigma)
        if np.sum(isin_interest) == 0:
            # 1つも興味がない場合, -1を返す
            return -1
        
        dist = np.exp(-dist)# 興味の範囲に入っていないものの距離を遠くする

        select_probs = np.exp(dist) * isin_interest
        select_probs /= select_probs.sum() # (candidate_num, )

        assert np.sum(select_probs == np.nan) == 0
        
        select_item = np.random.choice(len(select_probs),p=select_probs)
        
        self._update_state(item_embs[select_item, :])
        
        return select_item


if __name__ == "__main__":
    user = User(10, 0.01, 0.1)
    for i in range(5):
        rec_item_embs = np.random.normal(loc=0, scale=0.1, size=5*10).reshape(5, -1)
        print(user.response(rec_item_embs))
    
    
        