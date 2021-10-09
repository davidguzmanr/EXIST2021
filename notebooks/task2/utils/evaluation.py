from .preprocessing import preprocess

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DataSetText:    
    """
    Dataset for the texts.
    """
    def __init__(self, file):
        self.y = torch.tensor(file['label'].to_numpy(), dtype=torch.long)
        self.texts = file['text'].tolist()
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        
    def __getitem__(self, i):
        inputs1 = self.tokenizer(self.texts[i], max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        return torch.squeeze(inputs1['input_ids']), torch.squeeze(inputs1['attention_mask']), torch.squeeze(self.y[i])
        
    def __len__(self):
        return len(self.y)

class SexismClassifier(nn.Module):
    """
    Class for the model trained for task2.
    """
    def __init__(self):
        super(SexismClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        
        self.dropout = nn.Dropout()
                
        self.clf = nn.Sequential(nn.Linear(768, 5), 
                                 nn.Softmax(dim=1))
        
    def forward(self, ids, masks):
        # [batch_size, max_length] -> [batch_size, max_length, 768]
        bert_output = self.bert(input_ids=ids, attention_mask=masks, return_dict=True).last_hidden_state
        
        # [batch_size, max_length, 768] -> [batch_size, 768]
        # cls_output = bert_output[:,0,:]
        cls_output = torch.mean(bert_output, dim=1)
        
        cls_output = self.dropout(cls_output)
        
        # [batch_size, 768] -> [batch_size, 6]
        y = self.clf(cls_output)

        return y

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
    for ids, masks, y_true in tqdm(loader_test):
        ids = ids.to(device)
        masks = masks.to(device)
        y_true = y_true.to(device)

        y_hat = model(ids, masks)

        loss = F.cross_entropy(y_hat, y_true)

        y_pred = torch.argmax(y_hat, dim=1)

        ys.extend(y_true.cpu().numpy().tolist() )
        ys_hat.extend(y_pred.cpu().numpy().tolist())
        
    return ys, ys_hat


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