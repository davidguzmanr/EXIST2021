import streamlit as st

import os

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

input_text = st.text_input(label='Text', max_chars=150)

