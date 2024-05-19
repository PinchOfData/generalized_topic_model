#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from patsy import dmatrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gtm.utils import bert_embeddings_from_list
from copy import deepcopy
import pandas as pd
class GTMCorpus(Dataset):
  """
  Corpus for the GTM model.
  """

  def __init__(self, df, prevalence=None, content=None, prediction=None, labels=None, embeddings_type=None,
               count_words=True, normalize_doc_length=False, vectorizer=None, vectorizer_args={},
               sbert_model_to_load=None, batch_size=200, max_seq_length=100000,
               doc2vec_args={},):
    """
    Initialize GTMCorpus.

    Args:
        df : pandas DataFrame. Must contain a column 'doc' with the text of each document. If count_words=True, it must also contain 'doc_clean' with the cleaned text of each document.
        prevalence : string, formula for prevalence covariates (of the form "~ cov1 + cov2 + ..."), but allows for transformations of e.g., "~ g(cov1) + h(cov2) + ...)". Use "C(your_categorical_variable)" to indicate a categorical variable. See the Patsy package for more details.
        content : string, formula for content covariates (of the form "~ cov1 + cov2 + ..."). Use "C(your_categorical_variable)" to indicate a categorical variable. See the Patsy package for more details.
        prediction : string, formula for covariates used as inputs for the prediction task (also of the form "~ cov1 + cov2 + ..."). See the Patsy package for more details.
        labels : string, formula for labels used as outcomes for the prediction task (of the form "~ label1 + label2 + ...")
        embeddings_type : (optional) string, type of embeddings to use. Can be 'Doc2Vec' or 'SentenceTranformer'
        count_words : boolean, whether to produce a document-term matrix or not
        normalize_doc_length : boolean, whether to normalize the document-term matrix by document length (to accomodate for varying document lengths)
        vectorizer_args: dict, arguments for the CountVectorizer object
        vectorizer : sklearn CountVectorizer object, if None, a new one will be created
        sbert_model_to_load : string, name of the SentenceTranformer model to load
        batch_size : int, batch size for SentenceTranformer embeddings
        max_seq_length : int, maximum sequence length for SentenceTranformer embeddings
    """

    # Basic params and formulas
    self.prevalence = prevalence
    self.content = content
    self.prediction = prediction
    self.labels = labels
    self.embeddings_type = embeddings_type
    self.count_words = count_words
    self.count_vectorizer_args = vectorizer_args
    self.normalize_doc_length = normalize_doc_length
    self.vectorizer = vectorizer
    self.sbert_model_to_load = sbert_model_to_load
    self.batch_size = batch_size
    self.max_seq_length = max_seq_length
    self.doc2vec_args = doc2vec_args
    self.df = df

    # concat the column doc_clean and doc_clean_translated
    self.doc_concated = list(pd.concat([df['doc_clean'], df['doc_clean_translated']]))
    # number of rows in doc_clean
    N_doc = len(df['doc_clean'])

    # Compute bag of words matrix
    if self.count_words:
      print('Computing bag of words matrix')
      if vectorizer is None:
        self.vectorizer = CountVectorizer(**vectorizer_args)
      else:
        self.vectorizer = vectorizer
      #TODO: construct M_bow_original and M_bow_translated
      self.M_bow = self.vectorizer.fit_transform(self.doc_concated)

      if normalize_doc_length:
        self.M_bow = self.M_bow / self.M_bow.sum(axis=0)

      self.M_bow_original = self.M_bow[:N_doc]
      self.M_bow_translated = self.M_bow[N_doc:]
      self.vocab = self.vectorizer.get_feature_names_out()
      self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}
      self.log_word_frequencies = torch.FloatTensor(np.log(np.array(self.M_bow.sum(axis=0)).flatten()))
    else:
      self.M_bow = None
      self.vocab = None
      self.id2token = None

    #TODO: Create embeddings matrix
    self.M_embeddings = None
    self.V_embeddings = None
    self.M_embeddings_original = None
    self.M_embeddings_translated = None

    if embeddings_type == 'SentenceTransformer':
      print('Computing SentenceTransformer embeddings')
      self.M_embeddings = bert_embeddings_from_list(self.doc_concated, sbert_model_to_load, batch_size, max_seq_length)
      self.V_embeddings = bert_embeddings_from_list(self.vocab, sbert_model_to_load, batch_size, max_seq_length)

      self.M_embeddings_original = self.M_embeddings[:N_doc]
      self.M_embeddings_translated = self.M_embeddings[N_doc:]

    # Extract prevalence covariates matrix
    if prevalence is not None:
      self.prevalence_colnames, self.M_prevalence_covariates = self._transform_df(prevalence)
    else:
      self.M_prevalence_covariates = np.zeros((len(df.index), 1), dtype=np.float32)

    # Extract content covariates matrix
    if content is not None:
      self.content_colnames, self.M_content_covariates = self._transform_df(content)
    else:
      self.M_content_covariates = None

    # Extract prediction covariates matrix
    if prediction is not None:
      self.prediction_colnames, self.M_prediction = self._transform_df(prediction)
    else:
      self.M_prediction = None

    # Extract labels matrix
    # if labels is not None:
    #   self.labels_colnames, self.M_labels = self._transform_df(labels)
    # else:
    #   self.M_labels = None

    if labels is not None:
      self.M_labels = df[labels].values
    else:
      self.M_labels = None

    language_encoded = df['language'].apply(lambda x: 0 if x == 'en' else 1)
    self.language = pd.DataFrame(language_encoded, columns=['language'])
    # make it a numpy array
    self.language = np.array(self.language).squeeze()

  def _transform_df(self, formula):
    """
    Uses the patsy package to transform covariates into appropriate matrices
    """

    M = dmatrix(formula, self.df)
    colnames = M.design_info.column_names
    M = np.asarray(M, dtype=np.float32)

    return colnames, M

  def __len__(self):
    """Return length of dataset."""
    return len(self.df)

  def __getitem__(self, i):
    """Return sample from dataset at index i"""
    #TODO: we have two modes
    # mode 1: original as input, and translated as output
    # mode 2: translated as input, and original as output
    # choose mode randomly
    mode = np.random.randint(2)
    mode = 'original_to_translated' if mode == 0 else 'translated_to_original'

    d = {}
    M_bow_sample_original = None
    M_bow_sample_translated = None
    if self.M_bow is not None:
      if type(self.M_bow_original[i]) == scipy.sparse.csr_matrix:
        M_bow_sample_original = torch.FloatTensor(self.M_bow_original[i].todense())
      else:
        M_bow_sample_original = torch.FloatTensor(self.M_bow_original[i])

      if type(self.M_bow_translated[i]) == scipy.sparse.csr_matrix:
        M_bow_sample_translated = torch.FloatTensor(self.M_bow_translated[i].todense())
      else:
        M_bow_sample_translated = torch.FloatTensor(self.M_bow_translated[i])

    if mode == 'original_to_translated':
      d["M_bow_input"] = M_bow_sample_original
      d["M_bow_output"] = M_bow_sample_translated

      if self.M_embeddings is not None:
        d['M_embeddings_input'] = self.M_embeddings_original[i]
        d['M_embeddings_output'] = self.M_embeddings_translated[i]

    else:
      d["M_bow_input"] = M_bow_sample_translated
      d["M_bow_output"] = M_bow_sample_original

      if self.M_embeddings is not None:
        d['M_embeddings_input'] = self.M_embeddings_translated[i]
        d['M_embeddings_output'] = self.M_embeddings_original[i]

    if self.prevalence is not None:
      d['M_prevalence_covariates'] = self.M_prevalence_covariates[i]

    if self.content is not None:
      d['M_content_covariates'] = self.M_content_covariates[i]

    if self.prediction is not None:
      d['M_prediction'] = self.M_prediction[i]

    if self.M_labels is not None:
      d['M_labels'] = self.M_labels[i]

    if self.language is not None:
      d['language'] = self.language[i]

    return d
