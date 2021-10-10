import streamlit as st
import streamlit.components.v1 as components

import torch
import numpy as np

from transformers import BertTokenizer

from notebooks.task1.utils.evaluation import SexismClassifier as SexismClassifierTask1
from notebooks.task2.utils.evaluation import SexismClassifier as SexismClassifierTask2

from os import mkdir
from os.path import exists
import gdown
from lime.lime_text import LimeTextExplainer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
berttokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

@st.cache(allow_output_mutation=True)
def load_model_task1():
    model_path = 'notebooks/task1/models/sexism-classifier-task1.pt'
    if not exists(model_path):
        # Create directory
        try:
            mkdir('notebooks/task1/models/')
        except Exception as e:
            print(e)        

        # Download the model
        url = 'https://drive.google.com/uc?id=1V0VbdwXDcFP6f0GrdCna1SQqqkZpBOLW'
        gdown.download(url, model_path, quiet=False)
    
    model = SexismClassifierTask1()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    return model

@st.cache(allow_output_mutation=True)
def load_model_task2():
    model_path = 'notebooks/task2/models/sexism-classifier-task2.pt'
    if not exists(model_path):
        # Create directory
        try:
            mkdir('notebooks/task2/models/')
        except Exception as e:
            print(e)

        # Download the model
        url = 'https://drive.google.com/uc?id=1AtE9iu5OWeTpYTeMa_xrCvsmSuGVyFdJ'
        gdown.download(url, model_path, quiet=False)
    
    model = SexismClassifierTask2()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    return model
    
def prepape_text(text):
    berttokenizer_dict = berttokenizer(text.lower(), max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = berttokenizer_dict['input_ids'].to(device)
    attention_mask = berttokenizer_dict['attention_mask'].to(device)

    return input_ids, attention_mask

def make_prediction_task1(text):
    id_to_label = {0: 'not-sexist', 1: 'sexist'}

    input_ids, attention_mask = prepape_text(text)
    prediction = model_task1(input_ids, attention_mask)
    prediction = prediction.detach().cpu().numpy().squeeze()
    
    y_pred = prediction.argmax()
    y_label = id_to_label[y_pred]
    y_prob = prediction[y_pred]

    return {'label': y_label, 'probability': y_prob}

def explain_predictions_task1(string_list):
    predictions = []
    for string in string_list:
        input_ids, attention_mask = prepape_text(string)
        prediction = model_task1(input_ids, attention_mask)
        prediction = prediction.detach().cpu().numpy().squeeze()
        predictions.append(prediction)

    return np.array(predictions)

def make_prediction_task2(text):
    id_to_label = {0: 'ideological-inequality', 1: 'misogyny-non-sexual-violence', 2: 'objectification', 
                   3: 'sexual-violence', 4: 'stereotyping-dominance'}

    input_ids, attention_mask = prepape_text(text)
    prediction = model_task2(input_ids, attention_mask)
    prediction = prediction.detach().cpu().numpy().squeeze()
    
    y_pred = prediction.argmax()
    y_label = id_to_label[y_pred]
    y_prob = prediction[y_pred]

    return {'label': y_label, 'probability': y_prob}

description = """
*Model to classify and categorize sexism for the [EXIST: sEXism Identification in Social neTworks](http://nlp.uned.es/exist2021/) competition.*
"""

st.sidebar.markdown(description)

"""
# EXIST: sEXism Identification in Social neTworks
"""

text = """
<div style="text-align: justify"> 
The Oxford English Dictionary defines sexism as <b>“prejudice, stereotyping or discrimination, typically against women, on the basis of sex”</b>. 
Inequality and discrimination against women that remain embedded in society is increasingly being replicated online.
<br><br>
Detecting online sexism may be difficult, as it may be expressed in very different forms. Sexism may sound “friendly”: 
the statement <i>“Women must be loved and respected, always treat them like a fragile glass”</i> may seem positive, 
but is actually considering that women are weaker than men. Sexism may sound “funny”, as it is the case of sexist jokes 
or humour (<i>“You have to love women… just that… You will never understand them.”</i>). Sexism may sound “offensive” and 
“hateful”, as in <i>“Humiliate, expose and degrade yourself as the fucking bitch you are if you want a real man to give 
you attention”</i>. Our aim is the detection of sexism in a broad sense, from explicit misogyny to other subtle expressions 
that involve implicit sexist behaviours.
<br><br>
However, even the most subtle forms of sexism can be as pernicious as the most violent ones and affect women in many facets 
of their lives, including domestic and parenting roles, career opportunities, sexual image and life expectations, to name a few. 
The automatic identification of sexisms in a broad sense may help to create, design and determine the evolution of new equality policies, 
as well as encourage better behaviors in society.
<br><br>
</div>
"""

st.markdown(text, unsafe_allow_html=True)

"""
## TASK 1: Sexism Identification
"""

text = """
<div style="text-align: justify"> 
The first subtask is a binary classification. The systems have to decide whether or not a given text is sexist, i.e., 
it is sexist itself, describes a sexist situation or criticizes a sexist behaviour. 
<br><br>
</div>
"""

st.markdown(text, unsafe_allow_html=True)

"""
## TASK 2: Sexism Categorization
"""

text = """
<div style="text-align: justify"> 
Once a message has been classified as sexist, the second task aims to categorize the message according to the type of sexism 
(according to the categorization proposed by experts and that takes into account the different facets of women that are undermined). 
In particular, the competition proposes a five-classification task:

<ol type="1">
  <li><b>IDEOLOGICAL AND INEQUALITY:</b> The text discredits the feminist movement, rejects inequality between men and women, 
  or presents men as victims of gender-based oppression.</li>
  <li><b>STEREOTYPING AND DOMINANCE:</b> The text expresses false ideas about women that suggest they are more suitable to 
  fulfill certain roles (mother, wife, family caregiver, faithful, tender, loving, submissive, etc.), or inappropriate for 
  certain tasks (driving, hardwork, etc), or claims that men are somehow superior to women.</li>
  <li><b>OBJECTIFICATION:</b> The text presents women as objects apart from their dignity and personal aspects, or assumes 
  or describes certain physical qualities that women must have in order to fulfill traditional gender roles (compliance with 
  beauty standards, hypersexualization of female attributes, women’s bodies at the disposal of men, etc.).</li>
  <li><b>SEXUAL VIOLENCE:</b> Sexual suggestions, requests for sexual favors or harassment of a sexual nature (rape or 
  sexual assault) are made.</li>
  <li><b>MISOGYNY AND NON-SEXUAL VIOLENCE:</b> The text expressses hatred and violence towards women.</li>
</ol> 

</div>
"""

st.markdown(text, unsafe_allow_html=True)

"""
## Model
"""

text = """
<div style="text-align: justify"> 
I applied fine-tuning to <a href="https://huggingface.co/bert-base-multilingual-uncased" target="_blank">bert-base-multilingual-uncased</a>, 
which is a model pretrained on a large corpus of multilingual data in a self-supervised fashion, to make the classification. I chose this model
because the task contained text in English and Spanish.
<br><br>
Here you can try the best model I found (sorry for the text, it's just an example):
<br><br>
</div>
"""

st.markdown(text, unsafe_allow_html=True)

# Sorry for this, it is just to show it works
input_text = st.text_input(label='Text', value='You look like a whore with those jeans', max_chars=150)

model_task1 = load_model_task1()
model_task2 = load_model_task2()

prediction_task1 = make_prediction_task1(input_text)

st.write('**Task 1:**')
st.write(prediction_task1)

if prediction_task1['label'] == 'sexist':
    prediction_task2 = make_prediction_task2(input_text)

    st.write('**Task 2:**')
    st.write(prediction_task2)

explain_pred = st.button('Explain predictions')

if explain_pred:
    with st.spinner('Generating explanations'):
        class_names = ['not-sexist', 'sexist']
        explainer = LimeTextExplainer(class_names=class_names, feature_selection='none', random_state=42)
        exp = explainer.explain_instance(input_text, explain_predictions_task1, num_samples=1000)
        components.html(exp.as_html(), height=400, scrolling=False)

text = """
<div style="text-align: justify"> 
In the table below you can see the performance of my model, which is quite close to the winner of the shared task (see 
<a href="http://nlp.uned.es/exist2021/#results" target="_blank">EXIST-2021 results</a>). 
<br><br>
</div>
"""

st.markdown(text, unsafe_allow_html=True)

table = """
<style type="text/css">
.tg  {border:none;border-collapse:collapse;border-color:#ccc;border-spacing:0;margin:0px auto;}
.tg td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:0px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#f0f0f0;border-color:#ccc;border-style:solid;border-width:0px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-mxj2{background-color:#f9f9f9;border-color:inherit;font-style:italic;text-align:center;vertical-align:top}
.tg .tg-yynm{background-color:#f0f0f0;border-color:#000000;color:#333333;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-abip{background-color:#f9f9f9;border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-yynm" rowspan="2">Model</th>
    <th class="tg-7btt" colspan="2">Task 1: Sexism Identification</th>
    <th class="tg-7btt" colspan="2">Task 2: Sexism Categorization</th>
  </tr>
  <tr>
    <th class="tg-mxj2">Accuracy</th>
    <th class="tg-mxj2">F1</th>
    <th class="tg-mxj2">Accuracy</th>
    <th class="tg-mxj2">F1</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">AI-UPV (winner)</td>
    <td class="tg-c3ow">0.7804</td>
    <td class="tg-c3ow">0.7802</td>
    <td class="tg-c3ow">0.6577</td>
    <td class="tg-c3ow">0.5787</td>
  </tr>
  <tr>
    <td class="tg-abip">My model</td>
    <td class="tg-abip">0.7537</td>
    <td class="tg-abip">0.7519</td>
    <td class="tg-abip">0.6165</td>
    <td class="tg-abip">0.5308</td>
  </tr>
</tbody>
</table>
"""

st.markdown(table, unsafe_allow_html=True)