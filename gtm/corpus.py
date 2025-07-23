#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from patsy import dmatrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from typing import Optional, Dict
import pandas as pd

class GTMCorpus(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        modalities: Optional[Dict[str, Dict]] = None,
        prevalence: Optional[str] = None,
        content: Optional[str] = None,
        prediction: Optional[str] = None,
        labels: Optional[str] = None
    ):
        self.df = df

        # If no modalities provided, fall back to simple BoW on doc_clean
        if modalities is None:
            default_vectorizer = (
                CountVectorizer()
            )
            modalities = {
                "default": {
                    "column": "doc_clean",
                    "views": {
                        "bow": {
                            "type": "bow",
                            "vectorizer": default_vectorizer
                        }
                    }
                }
            }

        self.modalities_config = modalities
        self.processed_modalities = {}

        for modality_name, modality_info in modalities.items():
            column = modality_info.get("column", "doc")
            views = modality_info.get("views", {})

            self.processed_modalities[modality_name] = {}

            for view_name, view_config in views.items():
                view_type = view_config["type"]

                # Decide which column to use
                if "column" in view_config:
                    view_column = view_config["column"]
                else:
                    view_column = column

                if view_type == "bow":
                    vec = view_config.get("vectorizer", CountVectorizer())
                    
                    if hasattr(vec, "vocabulary_"):
                        # vectorizer is already fitted (i.e., from training set)
                        M = vec.transform(df[view_column])
                    else:
                        # first time fitting (i.e., on training set)
                        M = vec.fit_transform(df[view_column])

                    self.processed_modalities[modality_name][view_name] = {
                        "matrix": M,
                        "vectorizer": vec,
                        "type": "bow"
                    }

                elif view_type == "embedding":
                    if "embed_fn" not in view_config:
                        raise ValueError("Embedding view requires an 'embed_fn' key with a callable.")

                    embed_fn = view_config["embed_fn"]
                    texts = df[view_column].tolist()
                    M = embed_fn(texts)

                    if isinstance(M, list):
                        M = torch.stack([torch.tensor(e) for e in M])
                    elif isinstance(M, np.ndarray):
                        M = torch.tensor(M)
                    elif not isinstance(M, torch.Tensor):
                        raise TypeError("Output of 'embed_fn' must be list, numpy.ndarray, or torch.Tensor")

                    self.processed_modalities[modality_name][view_name] = {
                        "matrix": M,
                        "type": "embedding"
                    }

                elif view_type == "raw":
                    self.processed_modalities[modality_name][view_name] = {
                        "data": df[view_column].tolist(),
                        "type": "raw"
                    }

                else:
                    raise ValueError(f"Unsupported view type: {view_type}")

        # Covariates (unchanged)
        self.prevalence = prevalence
        self.content = content
        self.prediction = prediction
        self.labels = labels

        if prevalence is not None:
            self.prevalence_colnames, self.M_prevalence_covariates = self._transform_df(prevalence)
        else:
            self.prevalence_colnames = []
            self.M_prevalence_covariates = np.zeros((len(df), 1), dtype=np.float32)

        if content is not None:
            self.content_colnames, self.M_content_covariates = self._transform_df(content)
        else:
            self.content_colnames = []
            self.M_content_covariates = None

        if prediction is not None:
            self.prediction_colnames, self.M_prediction = self._transform_df(prediction)
        else:
            self.prediction_colnames = []
            self.M_prediction = None

        if labels is not None:
            self.labels_colnames, self.M_labels = self._transform_df(labels)
        else:
            self.labels_colnames = []
            self.M_labels = None

        self.id2token = {}

        for modality_name, views in self.processed_modalities.items():
            for view_name, info in views.items():
                if info["type"] == "bow":
                    vocab = info["vectorizer"].get_feature_names_out()
                    self.id2token[f"{modality_name}_{view_name}"] = {
                        i: token for i, token in enumerate(vocab)
                    }

    def _transform_df(self, formula):
        M = dmatrix(formula, self.df)
        colnames = M.design_info.column_names
        M = np.asarray(M, dtype=np.float32)
        return colnames, M

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        d = {"modalities": {}}

        for modality_name, views in self.processed_modalities.items():
            d["modalities"][modality_name] = {}
            for view_name, info in views.items():
                view_type = info["type"]

                if view_type == "bow":
                    row = info["matrix"][i]
                    row = row.toarray().squeeze(0) if scipy.sparse.issparse(row) else row
                    d["modalities"][modality_name][view_name] = torch.FloatTensor(row)

                elif view_type == "embedding":
                    d["modalities"][modality_name][view_name] = info["matrix"][i]

                elif view_type == "raw":
                    d["modalities"][modality_name][view_name] = info["data"][i]

        if self.prevalence is not None:
            d["M_prevalence_covariates"] = self.M_prevalence_covariates[i]
        if self.content is not None:
            d["M_content_covariates"] = self.M_content_covariates[i]
        if self.prediction is not None:
            d["M_prediction"] = self.M_prediction[i]
        if self.labels is not None:
            d["M_labels"] = self.M_labels[i]

        return d
