from textblob import TextBlob
import spacy
from spacy.symbols import nsubj, nsubjpass, auxpass
from fastpunct import FastPunct

nlp = spacy.load("en_core_web_sm")
example = "Autonomous cars shift insurance liability toward manufacturers"
example1 = "In an Oct. 19 review of \"The Misanthrope\" at Chicago's Goodman Theatre (\"Revitalized Classics Take the Stage in Windy City,\" Leisure & Arts), the role of Celimene, played by Kim Cattrall, was mistakenly attributed to Christina Haag. Ms. Haag plays Elianti."
fastpunct = FastPunct('en')

def sentiment(text):
    res = TextBlob(example).sentiment
    return res

def passive_active(text):
    doc = nlp(text)
    # https://github.com/JasonThomasData/NLP_sentence_analysis/blob/master/stanford_NLTK.py#L52
    # above explains auxpass is only way to detect passive voice
    # https://gist.github.com/armsp/30c2c1e19a0f1660944303cf079f831a: setting rules for passive voice
    nsubj, nsubjpass, auxpass = 0, 0, 0
    for entity in doc:
        if (entity.dep == auxpass) or (entity.dep == nsubjpass): #or (entity.tag == VBN):
            return True
    return False

def ungrammatical(text):
    corrected = fastpunct.punct([text])[0]
    orig_tokens = text.split(' ')[:-1]
    corrected_tokens = corrected.split(' ')[:-1]
    allow_threshold = (len(orig_tokens) // 10 or 1)
    wrong = 0
    if len(corrected_tokens) != len(orig_tokens):
        wrong += abs(len(corrected_tokens), len(orig_tokens))
    for idx in range(min(len(corrected_tokens), len(orig_tokens))):
        if corrected_tokens[idx] != orig_tokens[idx]:
            wrong += 1
    if wrong > allow_threshold:
        return True # ungrammatical
    return False # grammatical
"""
Average Sentence Length
Tone - (능동,수동 / 긍정,부정)
Ungrammatical Sentence Ratio
"""
ungrammatical("This is hard range cock!!~!~ goood.")
