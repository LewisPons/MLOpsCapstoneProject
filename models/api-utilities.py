from string import punctuation
from dateutil import parser
import email
from re import sub
from typing import List, Union, Tuple

import pandas as pd
import numpy as np
import spacy
import multiprocessing as mp
from sklearn.base import TransformerMixin, BaseEstimator
from setfit import SetFitModel

from fastapi import FastAPI

import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 \
    import Features, EntitiesOptions, KeywordsOptions

import warnings
warnings.filterwarnings('ignore')


'''
NOTE: In the terminal set the cache folder for hugging face model files to avid warnings and possible failures
    $ export TRANSFORMERS_CACHE=$HOME/Desktop/MLOpsBootcamp/MLOpsCapstoneProject/models/hugging-faces-models/pretrained
    -models 
    $ export TOKENIZERS_PARALLELISM=true
'''
nlp = spacy.load("en_core_web_sm")
app = FastAPI()

class TextPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 nlp = nlp,
                 n_jobs=1):
        """
        Text preprocessing transformer includes steps:
            1. Punctuation removal
            2. Stop words removal   
            3. Lemmatization

        nlp  - spacy model
        n_jobs - parallel jobs to run
        """
        self.nlp = nlp
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self
    

    def transform(self, X, *_):
        X_copy = X.copy()

        partitions = 1
        cores = mp.cpu_count()
        if self.n_jobs <= -1:
            partitions = cores
        elif self.n_jobs <= 0:
            return X_copy.apply(self._preprocess_text)
        else:
            partitions = min(self.n_jobs, cores)

        data_split = np.array_split(X_copy, partitions)
        pool = mp.Pool(cores)
        data = pd.concat(pool.map(self._preprocess_part, data_split))
        pool.close()
        pool.join()

        return data
    
    
    def _remove_punct(self, doc):
        return (t for t in doc if t.text not in punctuation)
    

    def _remove_stop_words(self, doc):
        return (t for t in doc if not t.is_stop)
    

    def _lemmatize(self, doc):
        return ' '.join(t.lemma_ for t in doc)
    

    def _preprocess_text(self, text):
        doc = self.nlp(text)
        removed_punct = self._remove_punct(doc)
        removed_stop_words = self._remove_stop_words(removed_punct)
        return self._lemmatize(removed_stop_words)
    
    
    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)
    
    

def parse_raw_email(raw_email_str: str) -> pd.DataFrame:
    """
    Parses email raw text into a dataframe
    """
    
    # TODO: Some files have encoding troubles, as there are asci characteres that raises troubles in the open() block
    # TODO: Fix the encoding characters trouble
    try:
        msg = email.message_from_string(raw_email_str)    

        if 'Cc' in msg:
            _cc = [sub('\s+','', msg['Cc']).split(',')] 
        else: 
            _cc = [np.nan]
            
        if 'Bcc' in msg:
            _bcc = [sub('\s+','', msg['Cc']).split(',')] 
        else: 
            _bcc = [np.nan]
            
        if 'To' in msg:
            _to = [sub('\s+','', msg['To']).split(',')]
        else:
            _to = [np.nan]
        
        attributes = {  
            "Message-ID": [msg["Message-ID"]],
            "Date": [msg["Date"]],
            "From": [sub('\s+','', msg['From']).split(',')],
            "To": _to,
            "Subject": [msg["Subject"]],
            "Cc": _cc,
            "Mime-Version": [msg["Mime-Version"]],
            "Content-Type": [msg["Content-Type"]],
            "Content-Transfer-Encoding": [msg["Content-Transfer-Encoding"]],
            "Bcc": _bcc,
            "X-From": [msg["X-From"]],
            "X-To": [msg["X-To"]],
            "X-cc": [msg["X-cc"]],
            "X-bcc": [msg["X-bcc"]],
            "X-Folder": [msg["X-Folder"]],
            "X-Origin": [msg["X-Origin"]],
            "X-FileName": [msg["X-FileName"]]
        }

        if msg.is_multipart():
            for part in email.get_payload():
                body = part.get_payload() 
        else:
            body = msg.get_payload() 
            
        attributes['body'] = body
        df = pd.DataFrame(attributes, columns=attributes.keys())
        return df
    except:
        pass



def change_date_type(dates: Union[pd.DataFrame, pd.Series]) -> List:
    """
    Formats string column into datetime object
    """
    column = []
    
    for date in dates:
        column.append(parser.parse(date).strftime("%d-%m-%Y %H:%M:%S"))
    
    series = pd.Series(column)
    return pd.to_datetime(series)


def str_to_list(row):
    """
    convert a string List into a List
    """
    row = str(row).strip("[]").replace("'","")
    return row


def parsed_email_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic email formatting and cleaning
    """
    
    df['Date'] = change_date_type(df['Date'])
    
    df['body'] = df['body'].str.replace('\n','').str.replace('\t','')
    
    df['To'] = df['To'].astype('str')\
        .str.replace('b','')\
        .apply(str_to_list)
        
    df['From'] = df['From'].astype('str')\
        .str.replace('b','')\
        .apply(str_to_list)
    
    return df


def normalize_input_email(email_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Normalizes input body using Spacy utilities
    '''
    Normalizer = TextPreprocessor(nlp, -1)
    email_df['body_normalized'] = Normalizer.transform(email_df['body'])
    
    return  email_df

def generate_clasification_in_raw_email(raw_email: str, model) -> Tuple[str, pd.DataFrame]:
    """
    Parses raw email and classifies it in spam or not spam.
    """
    parsed_email = parse_raw_email(raw_email)
    proccessed_email_df = parsed_email_processing(parsed_email)
    normalized_df = normalize_input_email(proccessed_email_df)
    
    prediction = model(normalized_df['body_normalized'].to_list())    
    spam_result = 'Spam' if prediction[0] == 1 else 'Not Spam'
    
    normalized_df['spam_prediction'] = prediction
    normalized_df['spam_result'] = spam_result
    
    
    result = f"""The result of the mail: \n{raw_email} \nThe Email was classified as: {spam_result}"""
    
    return (result, normalized_df)


def generate_clasification_in_body(email_body: str, model) -> Tuple[str, str]:
    """
    Classifies an email in spam or not, receives only the email body.
    """
    email_body_df = pd.DataFrame({'body': [email_body]})
    normalized_df = normalize_input_email(email_body_df)
    
    prediction = model(normalized_df['body_normalized'].to_list())    
    spam_result = 'Spam' if prediction[0] == 1 else 'Not Spam'   
    
    result = f"""Email: \n{email_body} \n\nThe Email was classified as: {spam_result}"""
    return (result, spam_result)

def obtain_sentiment_analysis(email_body: str) -> dict:
     """
     Obtains sentiment analysis of the email.
     """

     authenticator = IAMAuthenticator('{apiKey}')
     natural_language_understanding = NaturalLanguageUnderstandingV1(
      version='2022-04-07',
      authenticator=authenticator)

     natural_language_understanding.set_service_url('{url}')

     response = natural_language_understanding.analyze(
      text=email_body,
      features=Features(
       entities=EntitiesOptions(emotion=True, sentiment=True, limit=2),
       keywords=KeywordsOptions(emotion=True, sentiment=True,
                                limit=2))).get_result()

     return(response)

@app.post("/")
async def inference(email_body : str):
     print(email_body)

    # model = SetFitModel.from_pretrained("lewispons/email-classifiers", cache_dir='pretrained-models')
     model = SetFitModel.from_pretrained("lewispons/large-email-classifier", cache_dir='pretrained-models')
    
     """ Use this if the input of the API is the whole raw email
     raw_email = raw_emails_examples[-2]
     results = generate_clasification_in_raw_email(raw_email , model)
     """
    
     #Use this if the input of the API is only the email body
     #email_body = test_not_spam_bodys[2]
     results = generate_clasification_in_body(email_body, model)
     sentiment = obtain_sentiment_analysis(email_body)

     result_dict = {
      'spam': results[1],
      'entities' : sentiment['entities'],
      'keywords' : sentiment['keywords']
     }

     return(result_dict)
