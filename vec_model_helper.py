import pandas as pd
import numpy as np
import re
from ult import get_contractions
import Stemmer
stemmer = Stemmer.Stemmer('english')

#sklearn for transforming text
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer,HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils




def expandShort(sent):
    ''' Expands contractions 
        Like don't -> do not
    '''
    for word in sent.split():
        if word.lower() in get_contractions():
            sent = sent.replace(word, get_contractions()[word.lower()])
    return sent

def cleanText(sent):
    ''' Cleans the text'''
    sent = sent.replace("\\n","")            
    sent = sent.replace("\\xa0","") #magic space lol
    sent = sent.replace("\\xc2","") #space
    sent = re.sub(r"(@[A-Za-z]+)|([\t])", "",sent)
    sent = expandShort(sent.strip().lower())
    sent = re.sub(r'[^\w]', ' ', sent)
    sent = re.sub(r"(@[A-Za-z]+)|([^A-Za-z \t])", " ", sent)
    ws = [w for w in sent.strip().split(' ') if w is not ''] # remove double space
    return " ".join(ws)


def stem(s):
    ''' Stemming using snowball'''
    ws = s.split(' ')
    ws = stemmer.stemWords(ws)
    return " ".join(ws)

def getVectorizer(num_features):
    hashingVectorizer = HashingVectorizer(n_features=num_features,alternate_sign=False, decode_error='strict')
    # tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    # vectorizer = Pipeline([("hashing", hashingVectorizer), ("tidf", tfidf)])
    return hashingVectorizer

def vectorizeBatch(batch,input_dim):
    num_features = input_dim
    v = getVectorizer(num_features)
    if v is None:
        v = getVectorizer(num_features)
    batch_vec = v.fit_transform(batch)
    return batch_vec

def encode_labels(Y):
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # one hot
    final_y = np_utils.to_categorical(encoded_Y)
    return final_y
    

def getLabel(batch):
    x = batch['Message']
    y = encode_labels(batch['Label'])
    return x,y

def get_batch_data(path, batch_size, input_dim):
    for batch in pd.read_csv(path, chunksize=batch_size):
        #Proccess batch and pass to fit_generator
        yield proccess_batch(batch, input_dim)
            
def proccess_batch(batch):
    text,labels = getLabel(batch)
    cleaned_batch = text.apply(lambda sentence: cleanText(sentence))
    cleaned_batch = cleaned_batch.apply(lambda sentence: stem(sentence))
    ###vectorized_batch = vectorizeBatch(cleaned_batch, input_dim) 
    #_lables = labels.reshape(labels.shape[0],1)
    return cleaned_batch, labels



