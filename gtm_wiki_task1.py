import pandas as pd
import sys
from gtm.corpus import GTMCorpus
from gtm.gtm_customized import GTM
import pickle as p
import os

def load_examples(language='en'):
  df = pd.read_csv('./data/wiki_shorts/{}/corpus/docs.txt'.format(language), header=None, delimiter='\t')
  df.columns = ['doc_clean']
  df['language'] = language
  # df = df.head(1000)
  return df

def create_dataset(language='en'):
  if not os.path.exists('train_dataset_intfloat_multilingual-e5-large-{}.pkl'.format(language)):
    print('Loading examples for {}'.format(language))
    df = load_examples(language)
    train_dataset = GTMCorpus(
      df,
      embeddings_type='SentenceTransformer',
      vectorizer_args = {'ngram_range':(1, 1), 'max_df':0.99, 'min_df':0.001, 'stop_words':'english'},
      sbert_model_to_load='intfloat/multilingual-e5-large',
      content=None,
      prevalence=None,
      batch_size=128,
      max_seq_length=512)
    print('Saving train_dataset_intfloat_multilingual-e5-large-{}.pkl'.format(language))
    with open('train_dataset_intfloat_multilingual-e5-large-{}.pkl'.format(language), 'wb') as f:
      p.dump(train_dataset, f)
  else:
    with open('train_dataset_intfloat_multilingual-e5-large-{}.pkl'.format(language), 'rb') as f:
      train_dataset = p.load(f)
  return train_dataset

train_dataset_en = create_dataset('en')
train_dataset_zh = create_dataset('zh')
