from spellchecker import SpellChecker
from nltk.tokenize import TweetTokenizer
import re

spell_es = SpellChecker(language='es')
spell_en = SpellChecker(language='en')

# See https://www.nltk.org/_modules/nltk/tokenize/casual.html#TweetTokenizer
# preserve_case=False -> possibly alter the case, but avoid changing emoticons like :D into :d
# reduce_len=True -> replace repeated character sequences of length 3 or greater with sequences of length 3
# strip_handles=True -> remove Twitter username handles from text
tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)


def spell_check(text, language):
    """
    Parameters
    ----------
    text: str.
        Text to spell check.
    language: str.
        'en' for english or 'es' for spanish.
    ----------

    Returns: str, corrected text.

    """

    # Only english or spanish for the moment
    assert language == 'en' or language == 'es'

    if language == 'en':
        # Separamos por signos de puntuación y espacios en blanco    
        sentence_split = re.split('([!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\s\t])', text)
        corrected_words = [spell_en.correction(word) for word in sentence_split if (word != ' ' and word != '')]
        corrected_sentence = ' '.join(corrected_words)
    else:
        # Separamos por signos de puntuación y espacios en blanco    
        sentence_split = re.split('([!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\s\t])', text)
        corrected_words = [spell_es.correction(word) for word in sentence_split if (word != ' ' and word != '')]
        corrected_sentence = ' '.join(corrected_words)
    
    return corrected_sentence


def preprocess(text):
    """
    Parameters
    ----------
    text: str.
        Text to spell check.
    ----------

    Returns: str, preprocessed text.

    """

    # Remove urls
    url_pattern = r'http\S+'
    text = re.sub(url_pattern, '', text)

    # Lowercase, remove repetitions and remove usernames
    text = tweet_tokenizer.tokenize(text)
    
    # Join everything
    text = ' '.join(text)

    # Sometimes all the text is just an url
    if text == '' or text == ' ':
        text = 'url'
    
    return text

