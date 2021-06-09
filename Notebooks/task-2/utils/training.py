from .preprocessing import preprocess

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

from os import listdir
from os.path import isfile, join
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary

from transformers import BertTokenizer, BertModel

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def infer(model, loader_test):
    """
    Returns the prediction of a model in a dataset.

    Parameters
    ----------
    model: PyTorch model
    
    loader_test: PyTorch DataLoader.

    Returns
    -------
    tuple
        y_true and y_pred 
    """
    model.eval()
    
    ys, ys_hat = [], []
    for ids, masks, y_true in loader_test:
        ids = ids.to(device)
        masks = masks.to(device)
        y_true = y_true.to(device)

        y_hat = model(ids, masks)

        loss = F.cross_entropy(y_hat, y_true)

        y_pred = torch.argmax(y_hat, dim=1)

        ys.extend(y_true.cpu().numpy().tolist() )
        ys_hat.extend(y_pred.cpu().numpy().tolist())
        
    return ys, ys_hat

def test(model, loader_test):
    """
    Returns the accuracy and loss of a model in a dataset.

    Parameters
    ----------
    model: PyTorch model
    
    loader_test: PyTorch DataLoader.

    Returns
    -------
    tuple
        Accuracy and loss of the model.
    """
    model.eval()
    
    f1_scores = []
    epochs_loss = []
    for ids, masks, y_true in loader_test:
        ids = ids.to(device)
        masks = masks.to(device)
        y_true = y_true.to(device)
        
        y_hat = model(ids, masks)        
        y_pred = torch.argmax(y_hat, 1)

        loss = F.cross_entropy(y_hat, y_true)
        epochs_loss.append(loss.item())
        
        f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')        
        f1_scores.append(f1)
        
        
    return np.mean(f1_scores), np.mean(epochs_loss)

def plot_history(losses_train, losses_test, f1_scores_train, f1_scores_test, epoch):
    """
    Plots the losses and accuracies of the model.

    Parameters
    ----------
    losses_train: array

    losses_test: array

    accs_train: array

    accs_test: array

    Returns
    -------
    None
    """

    fig, (ax0, ax1) = plt.subplots(figsize=(15,6), nrows=1, ncols=2)

    ax0.plot(np.arange(1, epoch+2), losses_train, marker='o', label='Train')
    ax0.plot(np.arange(1, epoch+2), losses_test, marker='o', label='Test')
    ax0.set_xlabel('Epochs', weight='bold')
    ax0.set_ylabel('Loss (cross entropy)', weight='bold')
    ax0.legend()

    ax1.plot(np.arange(1, epoch+2), f1_scores_train, marker='o', label='Train')
    ax1.plot(np.arange(1, epoch+2), f1_scores_test, marker='o', label='Test')
    ax1.set_xlabel('Epochs', weight='bold')
    ax1.set_ylabel('F1 score', weight='bold')
    ax1.legend()

    plt.show()
        

def train(model, loader_train, loader_test, epochs, lr=1e-5):
    """
    Trains a model.

    Parameters
    ----------
    model: PyTorch model.
    
    loader_train: PyTorch DataLoader.
    
    loader_test: PyTorch DataLoader.
    
    epochs: int
        Number of epochs to train the model

    lr: float, default=1e-5
        Learning rate for the optimizer.

    Returns
    -------
    tuple
        Tuple with the accuracy in the train and test dataset for each epoch.
    """
    opt = optim.Adam(model.parameters(), lr=lr)
    
    f1_scores_train, f1_scores_test = [], []   
    losses_train, losses_test = [], [] 
    for epoch in range(epochs):
        model.train()
        
        f1_scores, losses = [], []
        for ids, masks, y_true in tqdm(loader_train):
            opt.zero_grad()

            ids = ids.to(device)
            masks = masks.to(device)
            y_true = y_true.to(device)
            
            y_hat = model(ids, masks)
            
            loss = F.cross_entropy(y_hat, y_true)
            
            y_pred = torch.argmax(y_hat, dim=1)

            f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
            f1_scores.append(f1)
                        
            loss.backward()
            losses.append(loss.item())
            opt.step()
        
        f1_score_train = np.mean(f1_scores)
        loss_train = np.mean(losses)
        f1_scores_train.append(f1_score_train)
        losses_train.append(loss_train)
        
        f1_score_test, loss_test = test(model, loader_test)
        f1_scores_test.append(f1_score_test)
        losses_test.append(loss_test)
        
        clear_output(wait=True)
        print(f'Epoch {epoch + 1}: train F1:{f1_score_train} test F1:{f1_score_test}')
        plot_history(losses_train, losses_test, f1_scores_train, f1_scores_test, epoch)
        
    return losses_train, losses_test, f1_scores_train, f1_scores_test

berttokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
def predict_sexism(model, text, label=True):
    """
    Predict if there is sexism present in the text.

    Parameters
    ----------
    model: PyTorch model.

    text: str
        Text to analyze.

    label: boolean, default=False
        If True if will return the name of the label instead of the index.

    Returns:
    int or str
        Index or label of the prediction.  
    """
    labels_dict = {0: 'ideological-inequality', 1: 'misogyny-non-sexual-violence',
                   2: 'objectification', 3: 'sexual-violence', 4: 'stereotyping-dominance'}

    # Apply the same preprocessing
    text = preprocess(text)

    model.eval()

    berttokenizer_dict = berttokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = berttokenizer_dict['input_ids'].to(device)
    attention_mask = berttokenizer_dict['attention_mask'].to(device)

    if label:
        return labels_dict[torch.argmax(model(input_ids, attention_mask)).item()]     
    else:
        return torch.argmax(model(input_ids, attention_mask))      