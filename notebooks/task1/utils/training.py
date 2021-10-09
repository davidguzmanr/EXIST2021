from .preprocessing import preprocess

import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from transformers import BertTokenizer

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
    
    accs = []
    epochs_loss = []
    for ids, masks, y_true in loader_test:
        ids = ids.to(device)
        masks = masks.to(device)
        y_true = y_true.to(device)
        
        y_hat = model(ids, masks)
        
        y_pred = torch.argmax(y_hat, 1)
        
        acc = (y_true == y_pred).type(torch.float32).mean()
        
        accs.append(acc.item())

        loss = F.cross_entropy(y_hat, y_true)
        epochs_loss.append(loss.item())
        
    return np.mean(accs), np.mean(epochs_loss)

def plot_history(losses_train, losses_test, accs_train, accs_test, epoch):
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

    ax1.plot(np.arange(1, epoch+2), accs_train, marker='o', label='Train')
    ax1.plot(np.arange(1, epoch+2), accs_test, marker='o', label='Test')
    ax1.set_xlabel('Epochs', weight='bold')
    ax1.set_ylabel('Accuracy', weight='bold')
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
    
    accs_train, accs_test = [], []   
    losses_train, losses_test = [], [] 
    for epoch in range(epochs):
        model.train()
        
        accs, losses = [], []
        for ids, masks, y_true in tqdm(loader_train):
            opt.zero_grad()

            ids = ids.to(device)
            masks = masks.to(device)
            y_true = y_true.to(device)
            
            y_hat = model(ids, masks)
            
            loss = F.cross_entropy(y_hat, y_true)
            
            y_pred = torch.argmax(y_hat, dim=1)
            
            acc = (y_true == y_pred).type(torch.float32).mean()
            accs.append(acc.item())
            
            loss.backward()
            losses.append(loss.item())
            opt.step()
        
        acc_train = np.mean(accs)
        loss_train = np.mean(losses)
        accs_train.append(acc_train)
        losses_train.append(loss_train)
        
        acc_test, loss_test = test(model, loader_test)
        accs_test.append(acc_test)
        losses_test.append(loss_test)
        
        clear_output(wait=True)
        print(f'Epoch {epoch + 1}: train acc:{acc_train} test acc:{acc_test}')
        plot_history(losses_train, losses_test, accs_train, accs_test, epoch)
        
    return losses_train, losses_test, accs_train, accs_test

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
    labels_dict = {0: 'non-sexist', 1: 'sexist'}

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