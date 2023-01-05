import pickle
import os
HOME = os.environ['HOME']
import sys
dirs = "%s/work/program_016_inoue/master_programs"%HOME
sys.path.append(dirs)
#from pathlib import Path
#dirs = Path(dirs)

import pandas as pd

from sessrecs.models import GRU4Rec, GRUSQNRec, GRUSACRec, GRUBCQRec, GRUDQNRec, GRUQRDQNRec, GRUACRec
from sessrecs.trainer import BaseTrainer
from sessrecs.evaluator import evaluate

from sessrecs.dataset import MDPDataset, session_preprocess_data, split_data
from sessrecs.logger import create_logger, ResultLogger

import torch
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="GRU4Rec")
parser.add_argument("--datasetname", type=str, default="RC15")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--embed_dim", type=int, default=64)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--seq_len", type=int, default=10)
parser.add_argument("--preprocess", action="store_true")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--num_sessions", type=int, default=20000)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = create_logger("MLModel")

resultlogger = ResultLogger(file_path=os.path.join(dirs, "results/%s"%args.datasetname), k=args.k)

train, num_items = session_preprocess_data(
    datasetname=args.datasetname, 
    num_sessions=args.num_sessions,
    seq_len=args.seq_len, 
    logger=logger, 
    preprocess=args.preprocess
)

if args.model == "GRU4Rec":
    model = GRU4Rec(
        num_items,
        device=device,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim
    )
elif args.model == "GRUSAC":
    model = GRUSACRec(
        num_items,
        device=device,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim
    )
elif args.model == "GRUBCQ":
    model = GRUBCQRec(
        num_items,
        device=device,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim
    )
elif args.model == "GRUDQN":
    model = GRUDQNRec(
        num_items,
        device=device,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim
    )
elif args.model == "GRUSQN":
    model = GRUSQNRec(
        num_items,
        device=device,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim
    )
elif args.model == "GRUQRDQN":
    model = GRUQRDQNRec(
        num_items,
        device=device,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim
    )
elif args.model == "GRUAC":
    model = GRUACRec(
        num_items,
        device=device,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim
    )
else:
    raise ValueError("Not Existing model")

if args.preprocess:
    train, test = split_data(train, 0.8)
    valid, test = split_data(test, 0.5)
    pickle.dump(
        test,
        open("%s/work/dataset/%s/derived/test.df"%(HOME, args.datasetname), "wb"))
    pickle.dump(
        train,
        open("%s/work/dataset/%s/derived/train.df"%(HOME, args.datasetname), "wb"))
    pickle.dump(
        valid,
        open("%s/work/dataset/%s/derived/valid.df"%(HOME, args.datasetname), "wb"))
else:
    train = pickle.load(open("%s/work/dataset/%s/derived/train.df"%(HOME, args.datasetname), "rb"))
    valid = pickle.load(open("%s/work/dataset/%s/derived/valid.df"%(HOME, args.datasetname), "rb"))
    test = pickle.load(open("%s/work/dataset/%s/derived/test.df"%(HOME, args.datasetname), "rb"))
    logger.info("Loaded Data")

save_path = dirs +  "/models/%s/%s.pth"%(args.datasetname, model.__class__.__name__)

if not args.eval:
    train = MDPDataset(train)
    valid = MDPDataset(valid)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=True)


    trainer = BaseTrainer(
        model=model,
        logger=logger,
        #losslogger=losslogger,
        device=device
    )

    logger.info("%s Start Training"%model.__class__.__name__)

    trainer.fit(
        train_loader,
        valid_loader,
        args.epochs,
        save_path=save_path
    )

test = MDPDataset(test)
test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=True)

model.load(save_path)


results, evals, rec_results = evaluate(
    test_loader,
    model,
    device,
    k=args.k
)

print(evals)

for t in ["click", "purchase"]:
    hit = evals[t+"_Hit@%d"%args.k]
    ndcg = evals[t+"_NDCG@%d"%args.k]
    resultlogger.write_line(
        model.__class__.__name__,
        clicktype=t,
        hit=hit,
        ndcg=ndcg)

pd.DataFrame(rec_results).to_csv(dirs + "/results/%s/%s_rec_result.csv"%(args.datasetname, model.__class__.__name__), index=False)

results.to_csv(dirs + "/results/%s/%s_result.csv"%(args.datasetname, model.__class__.__name__))

logger.info("save results")
