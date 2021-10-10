# EXIST2021
My solution for [EXIST (sEXism Identification in Social neTworks)](http://nlp.uned.es/exist2021/).

First, create a virtual environment:

```
virtualenv exist2021-venv
```

To activate it:
```
source exist2021-venv/bin/activate
```

Then clone this repository and install the requirements:
```
git clone https://github.com/davidguzmanr/EXIST2021.git
cd EXIST2021
pip install -r requirements.txt
```

You can check the notebooks or check a little demo by running the app (it will take a few minutes to download the necessary files):

```
streamlit run app.py
```