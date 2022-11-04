import numpy as np
import random
import time
from numpy import vstack
from numpy import sqrt
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid, Tanh
from torch.nn import Module
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
from torch.utils.data.sampler import SubsetRandomSampler

from tqdm import tqdm

from metrics import get_metrics
from utils import SpaceTimeSplits, normalize_test_df_feat, normalize_train_df_feat
from constants import (
    TO_LOAD_CKPT,
    TRAINVAL_DATA_PATH,
    TRAIN_SPLIT_DATA_PATH,
    VAL_SPLIT_DATA_PATH,
    TEST_DATA_PATH,
    CHECKPOINT_PATH,
    LABEL,
    NARROWED_FEATS,
    GROUP_NAMES,
    TO_NORM_FEATS,
    TO_NORM_LABELS,
    IS_PRED_EXP,
    GRIDSEARCH_CV_SCORING,
    CV_FOLDS,
    NUM_EPOCHS,
    EVAL_EPOCH_EVERY,
    FEATS_TO_ENCODE,
)


pd.options.mode.chained_assignment = None   # to disable chained assignment warning (by default, this is "warn")

feats = NARROWED_FEATS

if torch.cuda.is_available():  
  dev = "cuda:0"
else:  
  dev = "cpu"
device = torch.device(dev)  


class CSVDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.X = df[feats].values.astype('float32')
        self.y = df[LABEL].values.astype('float32')
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 1))
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 10)
        self.norm1 = nn.LayerNorm(10)
        self.dropout1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(10, 8)
        self.norm2 = nn.LayerNorm(8)
        self.dropout2 = nn.Dropout(0.25)

        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = x.to(device)

        x = self.fc1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc4(x)

        return x

def prepare_data(train_df, val_df):
    train_dataset = CSVDataset(train_df)
    val_dataset = CSVDataset(val_df)
    
    # prepare data loaders
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    logger.debug(f"len(train_dl.dataset): {len(train_dl.dataset)}, len(val_dl.dataset):{len(val_dl.dataset)}")
    return train_dl, val_dl

def process_feats(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    encoded_feats = []
    for col in FEATS_TO_ENCODE:
        one_hot = pd.get_dummies(train_df[col])
        encoded_feats += list(one_hot.columns)
        train_df = train_df.drop(col,axis = 1)
        train_df = train_df.join(one_hot)

        one_hot = pd.get_dummies(val_df[col])
        encoded_feats += list(one_hot.columns)
        val_df = val_df.drop(col,axis = 1)
        val_df = val_df.join(one_hot)

        one_hot = pd.get_dummies(test_df[col])
        encoded_feats += list(one_hot.columns)
        test_df = test_df.drop(col,axis = 1)
        test_df = test_df.join(one_hot)

    train_df=train_df.reset_index()
    val_df=val_df.reset_index()
    test_df=test_df.reset_index()

    if TO_NORM_FEATS:
        for col_name in feats:
            # Normalize train features
            col_val, mu, sigma = normalize_train_df_feat(train_df, col_name)
            train_df.loc[:, (col_name)] = col_val

            # Normalize val features using mu and sigma from train
            col_val = normalize_test_df_feat(val_df, col_name, mu, sigma)
            val_df.loc[:, (col_name)] = col_val

            # Normalize test features using mu and sigma from train
            col_val = normalize_test_df_feat(test_df, col_name, mu, sigma)
            test_df.loc[:, (col_name)] = col_val
    if TO_NORM_LABELS:
        col_val, label_mu, label_sigma = normalize_train_df_feat(train_df, LABEL)
        train_df.loc[:, (LABEL)] = col_val

    return train_df, val_df, test_df


def train_model(train_dl, val_dl, model):
    criterion = MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0)

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        # print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS}")
        model.train()
        train_losses_epoch = 0
        for i, (inputs, targets) in enumerate(train_dl):
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
            train_losses_epoch += loss.item()

        train_losses.append(train_losses_epoch/len(train_dl))
        if (epoch+1)%EVAL_EPOCH_EVERY==0:
            print(f"Train MSE: {loss:.03f}")
            mse = evaluate_model(val_dl, model)
            val_losses.append(mse)
            print('Val MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))

    return train_losses, val_losses

def evaluate_model(val_dl, model):
    model.eval()

    with torch.no_grad():
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(val_dl):
            yhat = model(inputs)
            
            yhat = yhat.detach().cpu().numpy()
            actual = targets.cpu().numpy()
            actual = actual.reshape((len(actual), 1))
            
            predictions.append(yhat)
            actuals.append(actual)
        
        predictions, actuals = vstack(predictions), vstack(actuals)
        mse = mean_squared_error(actuals, predictions)
    return mse


def predict(row, model):
    model.eval()
    with torch.no_grad():
        row = Tensor([row])
        yhat = model(row)
        yhat = yhat.detach().cpu().numpy()
    return yhat


if __name__=="__main__":
    logger.debug(f"feats: {feats}")
    logger.debug(f"label: {LABEL}")
    start_time = time.time()

    train_df, val_df, test_df = process_feats(TRAIN_SPLIT_DATA_PATH, VAL_SPLIT_DATA_PATH, TEST_DATA_PATH)
    train_dl, val_dl = prepare_data(train_df, val_df)
    x_test, y_test = test_df[feats], test_df[LABEL]

    if TO_LOAD_CKPT:
        MODEL_PATH = f"{CHECKPOINT_PATH}/MLP_rmse60.66_epoch100.pt"
        model = MLP(len(feats))
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        model = model.to(device)
    else:
        model = MLP(len(feats))
        model = model.to(device)
        train_losses, val_losses = train_model(train_dl, val_dl, model)

        plt.plot(list(range(len(train_losses))), train_losses, label="train loss")
        val_loss_x = np.array(range(1,len(val_losses)+1))*10
        plt.plot(val_loss_x, val_losses, label="val loss")
        plt.legend()
        plt.show()
        plt.savefig("../output/train_val_loss.png")


    # make test set prediction
    yhat = predict(x_test.values.astype("float32"), model)
    yhat = yhat[0,:,0]
    if IS_PRED_EXP:
        test_metrics = get_metrics(y_test, yhat)
    else:
        test_metrics = get_metrics(y_test,  np.round(np.exp(yhat),2))

    # Display metrics
    test_metrics["source"] = "test set"
    print(test_metrics)
    print(f"Runtime: {(time.time()-start_time):.02f}s")

    if not TO_LOAD_CKPT:
        # Save trained model
        torch.save(
            model.state_dict(),
            f"{CHECKPOINT_PATH}/MLP_rmse{test_metrics['rmse']:.02f}_epoch{NUM_EPOCHS}.pt",
        )

