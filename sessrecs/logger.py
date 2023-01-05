from typing import Optional, Dict
#import tensorflow as tf
import os

from logging import Logger, getLogger, StreamHandler, DEBUG, Formatter

from csv import writer

def create_logger(
    name:Optional[str]=None
):
    logger = getLogger(name)
    logger.setLevel(DEBUG)
    logger.propagate = False
    
    ch = StreamHandler()
    ch.setLevel(DEBUG)
    ch.setFormatter(Formatter("%(asctime)s - %(filename)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)
    return logger

class ResultLogger(object):
    
    def __init__(self, file_path, k):
        self.file_path = os.path.join(file_path, "result@%d.csv"%k)
        self.k = k
        if not os.path.exists(self.file_path):
            self.write_line("Model", "ClickType", "Hit@%d"%k, "NDCG@%d"%k)
        
    def write_line(self, model, clicktype, hit, ndcg):
        tocsv_list = [model, clicktype, hit, ndcg]
        with open(self.file_path, "a", newline='') as f:
            writer_obj = writer(f)
            writer_obj.writerow(tocsv_list)
            f.close()


# class LossLogger(object):

#     def __init__(
#         self, 
#         logdir:str,
#         datasetname: str,
#         modelname : str
#     ):
#         self.logdir = logdir

#         self.summary_writer = tf.summary.create_file_writer(os.path.join(logdir, datasetname, modelname))
    
#     def write_loss(
#         self,
#         info:Dict[str, float],
#         episode:int
#     ):
#         with self.summary_writer.as_default():
#             for key in info.keys():
#                 tf.summary.scalar(key, info[key], step=episode)