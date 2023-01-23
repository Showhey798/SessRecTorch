import pickle
import os
HOME = os.environ['HOME']
import sys
DIRS = "%s/work"%HOME
sys.path.append(DIRS)

import pandas as pd

from sessrecs.models import GRU4Rec, GRUSQNRec, GRUSACRec, GRUBCQRec, GRUDQNRec, GRUQRDQNRec, GRUACRec, GRUCQLRec, GRUREMRec, BaseRecommender
from sessrecs.trainer import BaseTrainer
from sessrecs.evaluator import evaluate
from sessrecs.dataset import MDPDataset, session_preprocess_data
from sessrecs.logger import create_logger, log_params_from_omegaconf_dict

import torch
from torch.utils.data import DataLoader

import mlflow
import sqlite3

import hydra
from omegaconf import DictConfig

DB_PATH = os.path.join(DIRS, "logs/db/mlruns.db") # トラッキングサーバ
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH) # バックエンド用のDBを作成
ARTIFACT_LOCATION = os.path.join(DIRS, "logs/artifact/")

tracking_uri = f'sqlite:///{DB_PATH}'
mlflow.set_tracking_uri(tracking_uri)


@hydra.main(version_base=None, config_path="%s/work/conf/"%(os.environ["HOME"]), config_name="config")
def main(cfg:DictConfig):

    # mlflowの設定
    EXPERIMENT_NAME = cfg.model.name
    
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=ARTIFACT_LOCATION
        )
    else:
        experiment_id = experiment.experiment_id
        
    
    
    DATAPATH = "%s/dataset/%s/derived"%(DIRS, cfg.dataset.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = create_logger("MLModel")
    
    # データの用意
    train, num_items = session_preprocess_data(
        datasetname=cfg.dataset.name, 
        num_sessions=cfg.dataset.num_sessions,
        seq_len=cfg.dataset.seq_len, 
        logger=logger, 
    )
    train = pickle.load(open(os.path.join(DATAPATH, "train.df"), "rb"))
    valid = pickle.load(open(os.path.join(DATAPATH, "valid.df"), "rb"))
    test = pickle.load(open(os.path.join(DATAPATH, "test.df"), "rb"))
    logger.info("Loaded Data")
    
    # モデルを選択
    if cfg.model.name == "gru4rec":
        model = GRU4Rec(
            num_items,
            device=device,
            cfg=cfg
        )
    elif cfg.model.name == "grucql":
        model = GRUCQLRec(
            num_items,
            device=device,
            cfg=cfg
        )
    elif cfg.model.name == "gruqrdqn":
        model = GRUQRDQNRec(
            num_items,
            device=device,
            cfg=cfg
        )
    elif cfg.model.name == "grubcq":
        model = GRUBCQRec(
            num_items,
            device=device,  
            cfg=cfg
        )
    elif cfg.model.name == "grusac":
        model = GRUSACRec(
            num_items,
            device=device,
            cfg=cfg
        )
    elif cfg.model.name == "grurem":
        model = GRUREMRec(
            num_items,
            device=device,
            cfg=cfg
        )
    # elif cfg.model.name == "grusqn":
    #     model = GRUSQNRec(
    #         num_items,
    #         device=device,
    #         cfg=cfg
    #     )
    # elif cfg.model.name == "gruac":
    #     model = GRUACRec(
    #         num_items,
    #         device=device,
    #         cfg=cfg
    #     )
    # elif cfg.model.name == "grudqn":
    #     model = GRUDQNRec(
    #         num_items,
    #         device=device,
    #         cfg=cfg
    #     )
    else:
        raise ValueError("Not Existing model")

    save_path = os.path.join(DIRS, "models/%s/%s.pth"%(cfg.dataset.name, model.__class__.__name__))

    train = MDPDataset(train)
    valid = MDPDataset(valid)

    train_loader = DataLoader(train, batch_size=cfg.training.batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=cfg.training.batch_size, shuffle=True)

    trainer = BaseTrainer(
        model=model,
        logger=logger,
        device=device
    )


    with mlflow.start_run(experiment_id=experiment_id) as run:
        
        log_params_from_omegaconf_dict(cfg)
        logger.info("Training....")
        loss_hist, best_score = trainer.fit(
            train_loader,
            valid_loader,
            cfg.training.epochs,
            save_path=save_path
        )
        mlflow.log_metric("val_purchase_Hit_at_10", best_score)
        

        test = MDPDataset(test)
        test_loader = DataLoader(test, batch_size=cfg.training.batch_size, shuffle=True)

        model.load(save_path)

        results, evals, rec_results = evaluate(
            test_loader,
            model,
            device,
            k=10
        )
        
        mlflow.log_metrics(evals.to_dict())

        print(evals)

        #pd.DataFrame(rec_results).to_csv(os.path.join(DIRS, "/results/%s/%s_rec_result.csv"%(cfg.dataset.name, model.__class__.__name__)), index=False)

        #results.to_csv(os.path.join(DIRS, "/results/%s/%s_result.csv"%(cfg.dataset.name, model.__class__.__name__)))

        logger.info("save results")


if __name__ == "__main__":
    main()