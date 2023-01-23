from typing import Optional
import numpy as np

class ItemClusters():
    
    def __init__(
        self,
        num_cluster_items:int,
        embed_dim:int, # ユーザの興味座標の次元数
        mean:Optional[float]=0.,
        scale:Optional[float]=1.
    ):
        self.feature_embeddings = np.random.normal(
            loc=mean, scale=scale, size=num_cluster_items*embed_dim
            ).reshape(num_cluster_items, -1)
    
    def get_fewatures(
        self, 
        slate_ids:np.ndarray # (num_slates,)
    ):
        embs = self.feature_embeddings[slate_ids, :]
        return embs
    

    
    
        