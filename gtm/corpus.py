#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from patsy import dmatrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gtm.utils import bert_embeddings_from_list


class GTMCorpus(Dataset):
    """
    Corpus for the GTM model.
    """

    def __init__(
        self,
        df,
        prevalence=None,
        content=None,
        prediction=None,
        labels=None,
        embeddings_type=None,
        normalize_doc_length=False,
        vectorizer=None,
        vectorizer_args={},
        sbert_model_to_load=None,
        batch_size=64,
        max_seq_length=100000,
        doc2vec_args={},
        device=None,
        num_en=0,
        num_zh=0,
        language='en' # 'en' or 'zh' or 'both'
    ):
        """
        Initialize GTMCorpus.

        Args:
            df : pandas DataFrame. Must contain a column 'doc' with the text of each document. If count_words=True, it must also contain 'doc_clean' with the cleaned text of each document.
            prevalence : string, formula for prevalence covariates (of the form "~ cov1 + cov2 + ..."), but allows for transformations of e.g., "~ g(cov1) + h(cov2) + ...)". Use "C(your_categorical_variable)" to indicate a categorical variable. See the Patsy package for more details.
            content : string, formula for content covariates (of the form "~ cov1 + cov2 + ..."). Use "C(your_categorical_variable)" to indicate a categorical variable. See the Patsy package for more details.
            prediction : string, formula for covariates used as inputs for the prediction task (also of the form "~ cov1 + cov2 + ..."). See the Patsy package for more details.
            labels : string, formula for labels used as outcomes for the prediction task (of the form "~ label1 + label2 + ...")
            embeddings_type : (optional) string, type of embeddings to use. Can be 'Doc2Vec' or 'SentenceTranformer'
            normalize_doc_length : boolean, whether to normalize the document-term matrix by document length (to accomodate for varying document lengths)
            vectorizer : sklearn CountVectorizer object, if None, a new one will be created
            vectorizer_args: dict, arguments for the CountVectorizer object
            sbert_model_to_load : string, name of the SentenceTranformer model to load
            batch_size : int, batch size for SentenceTranformer embeddings
            max_seq_length : int, maximum sequence length for SentenceTranformer embeddings
            doc2vec_args : dict, arguments for the Doc2Vec object
            device : string, device to use for SentenceTranformer embeddings
        """

        # Basic params and formulas
        self.prevalence = prevalence
        self.content = content
        self.prediction = prediction
        self.labels = labels
        self.embeddings_type = embeddings_type
        self.count_vectorizer_args = vectorizer_args
        self.normalize_doc_length = normalize_doc_length
        self.vectorizer = vectorizer
        self.sbert_model_to_load = sbert_model_to_load
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.doc2vec_args = doc2vec_args
        self.df = df
        self.device = device
        self.num_en = num_en
        self.num_zh = num_zh
        self.language = language

        # Compute bag of words matrix
        if vectorizer is None:
            self.vectorizer = CountVectorizer(**vectorizer_args)
            self.M_bow = self.vectorizer.fit_transform(df["doc_clean"])
        else:
            self.vectorizer = vectorizer
            self.M_bow = self.vectorizer.transform(df["doc_clean"])
        if normalize_doc_length:
            self.M_bow = self.M_bow / self.M_bow.sum(axis=0)
        self.vocab = self.vectorizer.get_feature_names_out()
        self.id2token = {
            k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)
        }
        self.log_word_frequencies = torch.FloatTensor(
            np.log(np.array(self.M_bow.sum(axis=0)).flatten())
        )

        # Create embeddings matrix
        self.M_embeddings = None
        self.V_embeddings = None

        # if embeddings_type == "Doc2Vec":
        #     clean_docs = [doc.split() for doc in df["doc_clean"]]
        #     tagged_documents = [
        #         TaggedDocument(doc, [i]) for i, doc in enumerate(clean_docs)
        #     ]
        #     self.Doc2Vec_model = Doc2Vec(tagged_documents, **doc2vec_args)
        #     self.M_embeddings = np.array(
        #         [self.Doc2Vec_model.infer_vector(doc) for doc in clean_docs]
        #     )
        #     self.V_embeddings = np.array(
        #         [self.Doc2Vec_model.infer_vector([token]) for token in self.vocab]
        #     )

        if embeddings_type == "SentenceTransformer":
            self.M_embeddings = bert_embeddings_from_list(
                df["doc_clean"], sbert_model_to_load, batch_size, max_seq_length
            )
            self.V_embeddings = bert_embeddings_from_list(
                self.vocab, sbert_model_to_load, batch_size, max_seq_length
            )

        # Extract prevalence covariates matrix
        if prevalence is not None:
            self.prevalence_colnames, self.M_prevalence_covariates = self._transform_df(
                prevalence
            )
        else:
            self.M_prevalence_covariates = np.zeros(
                (len(df.index), 1), dtype=np.float32
            )

        # Extract content covariates matrix
        if content is not None:
            self.content_colnames, self.M_content_covariates = self._transform_df(
                content
            )
        else:
            self.M_content_covariates = None

        # Extract prediction covariates matrix
        if prediction is not None:
            self.prediction_colnames, self.M_prediction = self._transform_df(prediction)
        else:
            self.M_prediction = None

        # Extract labels matrix
        if labels is not None:
            self.labels_colnames, self.M_labels = self._transform_df(labels)
        else:
            self.M_labels = None

        self.post_process()

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
        if self.language == 'en':
            return self.num_en
        elif self.language == 'zh':
            return self.num_zh
        else:
            return self.num_en + self.num_zh

    def post_process(self):
        '''
        limit examples to specific language, English always comes the first
        '''
        if self.language == 'en':
            self.M_bow = self.M_bow[:self.num_en] if self.M_bow is not None else None
            self.M_embeddings = self.M_embeddings[:self.num_en] if self.M_embeddings is not None else None
            self.M_prevalence_covariates = self.M_prevalence_covariates[:self.num_en] if self.M_prevalence_covariates is not None else None
            self.M_content_covariates = self.M_content_covariates[:self.num_en] if self.M_content_covariates is not None else None
            self.M_prediction = self.M_prediction[:self.num_en] if self.M_prediction is not None else None
            self.M_labels = self.M_labels[:self.num_en] if self.M_labels is not None else None
        elif self.language == 'zh':
            self.M_bow = self.M_bow[self.num_en:] if self.M_bow is not None else None
            self.M_embeddings = self.M_embeddings[self.num_en:] if self.M_embeddings is not None else None
            self.M_prevalence_covariates = self.M_prevalence_covariates[self.num_en:] if self.M_prevalence_covariates is not None else None
            self.M_content_covariates = self.M_content_covariates[self.num_en:] if self.M_content_covariates is not None else None
            self.M_prediction = self.M_prediction[self.num_en:] if self.M_prediction is not None else None
            self.M_labels = self.M_labels[self.num_en:] if self.M_labels is not None else None
        else:
            return

    def __getitem__(self, i):
        """Return sample from dataset at index i"""

        d = {}

        if type(self.M_bow[i]) == scipy.sparse.csr_matrix:
            M_bow_sample = torch.FloatTensor(self.M_bow[i].todense())
        else:
            M_bow_sample = torch.FloatTensor(self.M_bow[i])
        d["M_bow"] = M_bow_sample

        if self.M_embeddings is not None:
            d["M_embeddings"] = self.M_embeddings[i]

        if self.prevalence is not None:
            d["M_prevalence_covariates"] = self.M_prevalence_covariates[i]

        if self.content is not None:
            d["M_content_covariates"] = self.M_content_covariates[i]

        if self.prediction is not None:
            d["M_prediction"] = self.M_prediction[i]

        if self.labels is not None:
            d["M_labels"] = self.M_labels[i]

        return d
