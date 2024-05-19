#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from gtm.autoencoders import AutoEncoderMLP
from gtm.predictors import Predictor
from gtm.priors import DirichletPrior, LogisticNormalPrior
from gtm.utils import compute_mmd_loss, top_k_indices_column


class GTM:
    """
    Wrapper class for the Generalized Topic Model.
    """

    def __init__(
        self,
        train_data=None,
        test_data=None,
        n_topics=20,
        doc_topic_prior="dirichlet",
        update_prior=False,
        alpha=0.1,
        prevalence_model_type = "LinearRegression",
        prevalence_model_args = {},
        tol=0.001,
        encoder_input="bow",
        encoder_hidden_layers=[1024,512],
        encoder_non_linear_activation="relu",
        encoder_bias=True,
        encoder_include_prevalence_covariates=True,
        decoder_hidden_layers=[512,1024],
        decoder_non_linear_activation="relu",
        decoder_bias=True,
        predictor_type=None,
        predictor_hidden_layers=[],
        predictor_non_linear_activation="relu",
        predictor_bias=True,
        num_epochs=1000,
        num_workers=None,
        batch_size=64,
        learning_rate=1e-4,
        dropout=0.2,
        print_every_n_epochs=1,
        print_every_n_batches=1000,
        print_topics=True,
        print_content_covariates=True,
        log_every_n_epochs=1000,
        patience=1,
        delta=0,
        w_prior=1,
        w_pred_loss=1,
        ckpt_folder="../ckpt",
        ckpt=None,
        device=None,
        seed=42,
    ):
        """
        Args:
            train_data: a GTMCorpus object
            test_data: a GTMCorpus object
            n_topics: number of topics
            doc_topic_prior: prior on the document-topic distribution. Either 'dirichlet' or 'logistic_normal'.
            update_prior: whether to update the prior at each epoch to account for prevalence covariates.
            alpha: parameter of the Dirichlet prior (only used if update_prior=False)
            prevalence_model_type: type of model to estimate the prevalence of each topic. Either 'LinearRegression', 'RidgeCV', 'MultiTaskLassoCV', and 'MultiTaskElasticNetCV'.
            prevalence_model_args: dictionary with the parameters for the GLM on topic prevalence.
            tol: tolerance threshold to stop the MLE of the Dirichlet prior (only used if update_prior=True)
            encoder_input: input to the encoder. Either 'bow' or 'embeddings'. 'bow' is a simple Bag-of-Words representation of the documents. 'embeddings' is the representation from a pre-trained embedding model (e.g. GPT, BERT, GloVe, etc.).
            encoder_hidden_layers: list with the size of the hidden layers for the encoder.
            encoder_non_linear_activation: non-linear activation function for the encoder.
            encoder_bias: whether to use bias in the encoder.
            decoder_hidden_layers: list with the size of the hidden layers for the decoder (only used with decoder_type='mlp').
            decoder_non_linear_activation: non-linear activation function for the decoder (only used with decoder_type='mlp').
            decoder_bias: whether to use bias in the decoder (only used with decoder_type='mlp').
            predictor_type: type of predictor. Either 'classifier' or 'regressor'. 'classifier' predicts a categorical variable, 'regressor' predicts a continuous variable.
            predictor_hidden_layers: list with the size of the hidden layers for the predictor.
            predictor_non_linear_activation: non-linear activation function for the predictor.
            predictor_bias: whether to use bias in the predictor.
            num_epochs: number of epochs to train the model.
            num_workers: number of workers for the data loaders.
            batch_size: batch size for training.
            learning_rate: learning rate for training.
            dropout: dropout rate for training.
            print_every_n_epochs: number of epochs between each print.
            print_every_n_batches: number of batches between each print.
            print_topics: whether to print the top words per topic at each print.
            print_content_covariates: whether to print the top words associated to each content covariate at each print.
            log_every_n_epochs: number of epochs between each checkpoint.
            patience: number of epochs to wait before stopping the training if the validation or training loss does not improve.
            delta: threshold to stop the training if the validation or training loss does not improve.
            w_prior: parameter to control the tightness of the encoder output with the document-topic prior. If set to None, w_prior is chosen automatically.
            w_pred_loss: parameter to control the weight given to the prediction task in the likelihood. Default is 1.
            ckpt_folder: folder to save the checkpoints.
            ckpt: checkpoint to load the model from.
            device: device to use for training.
            seed: random seed.

        References:
            - Nan, F., Ding, R., Nallapati, R., & Xiang, B. (2019). Topic modeling with wasserstein autoencoders. arXiv preprint arXiv:1907.12374.
        """

        if ckpt:
            self.load_model(ckpt)

        else:
            self.n_topics = n_topics
            self.topic_labels = ["Topic_{}".format(i) for i in range(n_topics)]
            self.doc_topic_prior = doc_topic_prior
            if self.doc_topic_prior not in ["dirichlet", "logistic_normal"]:
                raise ValueError(
                    "The doc_topic_prior parameter must be either 'dirichlet' or 'logistic_normal'."
                )

            self.update_prior = update_prior
            self.alpha = alpha
            self.prevalence_model_type = prevalence_model_type
            self.prevalence_model_args = prevalence_model_args
            self.tol = tol
            self.encoder_input = encoder_input
            if self.encoder_input not in ["bow", "embeddings"]:
                raise ValueError(
                    "The encoder_input parameter must be either 'bow' or 'embeddings'."
                )

            self.encoder_hidden_layers = encoder_hidden_layers
            self.encoder_non_linear_activation = encoder_non_linear_activation
            if self.encoder_non_linear_activation not in ["relu", "sigmoid", None]:
                raise ValueError(
                    "The encoder_non_linear_activation parameter must be either 'relu' or 'sigmoid' or None."
                )

            self.encoder_bias = encoder_bias
            self.encoder_include_prevalence_covariates = (
                encoder_include_prevalence_covariates
            )
            self.decoder_hidden_layers = decoder_hidden_layers
            self.decoder_non_linear_activation = decoder_non_linear_activation
            if self.decoder_non_linear_activation not in ["relu", "sigmoid", None]:
                raise ValueError(
                    "The decoder_non_linear_activation parameter must be either 'relu' or 'sigmoid' or None."
                )

            self.decoder_bias = decoder_bias
            self.predictor_type = predictor_type
            self.predictor_hidden_layers = predictor_hidden_layers
            self.predictor_non_linear_activation = predictor_non_linear_activation
            if self.predictor_non_linear_activation not in ["relu", "sigmoid", None]:
                raise ValueError(
                    "The predictor_non_linear_activation parameter must be either 'relu' or 'sigmoid' or None."
                )

            self.predictor_bias = predictor_bias
            self.device = device
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.num_epochs = num_epochs
            if num_workers is None: 
                self.num_workers = mp.cpu_count()
            else:
                self.num_workers = num_workers
            self.dropout = dropout

            self.print_every_n_epochs = print_every_n_epochs
            self.print_every_n_batches = print_every_n_batches
            self.print_topics = print_topics
            self.print_content_covariates = print_content_covariates
            self.log_every_n_epochs = log_every_n_epochs
            self.patience = patience
            self.delta = delta
            self.w_prior = w_prior
            self.w_pred_loss = w_pred_loss
            self.ckpt_folder = ckpt_folder

            if not os.path.exists(ckpt_folder):
                os.makedirs(ckpt_folder)

            self.seed = seed
            if seed is not None:
                torch.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                np.random.seed(seed)

            if device is None:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                # elif torch.backends.mps.is_available():
                #     self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")

            bow_size = train_data.M_bow.shape[1]
            self.bow_size = bow_size

            if train_data.prevalence is not None:
                prevalence_covariate_size = train_data.M_prevalence_covariates.shape[1]
                self.prevalence_colnames = train_data.prevalence_colnames
            else:
                prevalence_covariate_size = 0

            if train_data.content is not None:
                content_covariate_size = train_data.M_content_covariates.shape[1]
                self.content_colnames = train_data.content_colnames
            else:
                content_covariate_size = 0

            if train_data.prediction is not None:
                prediction_covariate_size = train_data.M_prediction.shape[1]
                self.prediction_colnames = train_data.prediction_colnames
            else:
                prediction_covariate_size = 0

            if train_data.labels is not None:
                labels_size = train_data.M_labels.shape[1]
                if predictor_type == "classifier":
                    n_labels = len(np.unique(train_data.M_labels))
                else:
                    n_labels = 1
            else:
                labels_size = 0

            if encoder_input == "bow":
                self.input_size = bow_size
            elif encoder_input == "embeddings":
                input_embeddings_size = train_data.M_embeddings.shape[1]
                self.input_size = input_embeddings_size

            self.content_covariate_size = content_covariate_size
            self.prevalence_covariate_size = prevalence_covariate_size
            self.labels_size = labels_size
            self.id2token = train_data.id2token

            if self.encoder_include_prevalence_covariates:
                encoder_dims = [self.input_size + prevalence_covariate_size]
            else:
                encoder_dims = [self.input_size]
            encoder_dims.extend(encoder_hidden_layers)
            encoder_dims.extend([n_topics])

            decoder_dims = [n_topics + content_covariate_size]
            decoder_dims.extend(decoder_hidden_layers)
            decoder_dims.extend([bow_size])

            self.AutoEncoder = AutoEncoderMLP(
                encoder_dims=encoder_dims,
                encoder_non_linear_activation=encoder_non_linear_activation,
                encoder_bias=encoder_bias,
                decoder_dims=decoder_dims,
                decoder_non_linear_activation=decoder_non_linear_activation,
                decoder_bias=decoder_bias,
                dropout=dropout,
            ).to(self.device)

            if doc_topic_prior == "dirichlet":
                self.prior = DirichletPrior(
                    prevalence_covariate_size,
                    n_topics,
                    alpha,
                    prevalence_model_args,
                    tol,
                    device=self.device,
                )
            elif doc_topic_prior == "logistic_normal":
                self.prior = LogisticNormalPrior(
                    prevalence_covariate_size,
                    n_topics,
                    prevalence_model_type,
                    prevalence_model_args,
                    device=self.device,
                )

            if labels_size != 0:
                predictor_dims = [n_topics + prediction_covariate_size]
                predictor_dims.extend(predictor_hidden_layers)
                predictor_dims.extend([n_labels])
                self.predictor = Predictor(
                    predictor_dims=predictor_dims,
                    predictor_non_linear_activation=predictor_non_linear_activation,
                    predictor_bias=predictor_bias,
                    dropout=dropout,
                ).to(self.device)

            if self.labels_size != 0:
                self.optimizer = torch.optim.Adam(
                    list(self.AutoEncoder.parameters())
                    + list(self.predictor.parameters()),
                    lr=self.learning_rate,
                )
            else:
                self.optimizer = torch.optim.Adam(
                    self.AutoEncoder.parameters(), lr=self.learning_rate
                )

            self.epochs = 0
            self.reconstruction_loss = None
            self.mmd_loss = None
            self.prediction_loss = None

            self.train(train_data, test_data)

    def train(self, train_data, test_data):
        """
        Train the model.
        """

        train_data_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        if test_data is not None:
            test_data_loader = DataLoader(
                test_data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

        counter = 0
        best_loss = np.Inf
        best_epoch = -1
        self.save_model("{}/best_model.ckpt".format(self.ckpt_folder))
        
        for epoch in range(self.epochs, self.num_epochs):
            training_loss = self.epoch(train_data_loader, validation=False)

            if test_data is not None:
                validation_loss = self.epoch(test_data_loader, validation=True)

            if (epoch + 1) % self.log_every_n_epochs == 0:
                save_name = f'{self.ckpt_folder}/GTM_K{self.n_topics}_{self.doc_topic_prior}_{self.predictor_type}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_{self.epochs+1}.ckpt'
                self.save_model(save_name)

            if self.update_prior:
                if self.doc_topic_prior == "dirichlet":
                    posterior_theta = self.get_doc_topic_distribution(train_data)
                    self.prior.update_parameters(
                        posterior_theta, train_data.M_prevalence_covariates
                    )
                else:
                    posterior_theta = self.get_doc_topic_distribution(
                        train_data, to_simplex=False
                    )
                    self.prior.update_parameters(
                        posterior_theta, train_data.M_prevalence_covariates
                    )

            # Stopping rule for the optimization routine
            if test_data is not None:
                if validation_loss + self.delta < best_loss:
                    best_loss = validation_loss
                    best_epoch = self.epochs
                    self.save_model("{}/best_model.ckpt".format(self.ckpt_folder))
                    counter = 0
                else:
                    counter += 1
            else:
                if training_loss + self.delta < best_loss:
                    best_loss = training_loss
                    best_epoch = self.epochs
                    self.save_model("{}/best_model.ckpt".format(self.ckpt_folder))
                    counter = 0
                else:
                    counter += 1

            if counter >= self.patience:
                print(
                    "\nEarly stopping at Epoch {}. Reverting to Epoch {}".format(
                        self.epochs + 1, best_epoch + 1
                    )
                )
                ckpt = "{}/best_model.ckpt".format(self.ckpt_folder)
                self.load_model(ckpt)
                break

            self.epochs += 1

    def epoch(self, data_loader, validation=False):
        """
        Train the model for one epoch.
        """
        if validation:
            self.AutoEncoder.eval()
            if self.labels_size != 0:
                self.predictor.eval()
        else:
            self.AutoEncoder.train()
            if self.labels_size != 0:
                self.predictor.train()

        epochloss_lst = []
        for iter, data in enumerate(data_loader):
            if not validation:
                self.optimizer.zero_grad()

            # Unpack data
            for key, value in data.items():
                data[key] = value.to(self.device)
            bows = data.get("M_bow", None)
            bows = bows.reshape(bows.shape[0], -1)
            embeddings = data.get("M_embeddings", None)
            prevalence_covariates = data.get("M_prevalence_covariates", None)
            content_covariates = data.get("M_content_covariates", None)
            prediction_covariates = data.get("M_prediction", None)
            target_labels = data.get("M_labels", None)
            if self.encoder_input == "bow":
                x_input = bows
            elif self.encoder_input == "embeddings":
                x_input = embeddings
            x_bows = bows

            # Get theta and compute reconstruction loss
            if self.encoder_include_prevalence_covariates:
                x_recon, theta_q = self.AutoEncoder(
                    x_input, prevalence_covariates, content_covariates
                )
            else:
                prevalence_covariates_bis = None
                x_recon, theta_q = self.AutoEncoder(
                    x_input,
                    prevalence_covariates_bis,
                    content_covariates
                )

            reconstruction_loss = F.cross_entropy(x_recon, x_bows)
            
            # Get prior on theta and compute regularization loss
            theta_prior = self.prior.sample(
                N=x_input.shape[0],
                M_prevalence_covariates=prevalence_covariates,
                epoch=self.epochs
            ).to(self.device)
            mmd_loss = compute_mmd_loss(theta_q, theta_prior, device=self.device)

            if self.w_prior is None:
                mean_length = torch.sum(x_bows) / x_bows.shape[0]
                vocab_size = x_bows.shape[1]
                w_prior = mean_length * np.log(vocab_size)
            else:
                w_prior = self.w_prior

            # Predict labels and compute prediction loss
            if target_labels is not None:
                predictions = self.predictor(theta_q, prediction_covariates)
                if self.predictor_type == "classifier":
                    target_labels = target_labels.squeeze().to(torch.int64)
                    prediction_loss = F.cross_entropy(predictions, target_labels)
                elif self.predictor_type == "regressor":
                    prediction_loss = F.mse_loss(predictions, target_labels)
            else:
                prediction_loss = 0

            # Total loss
            loss = (
                reconstruction_loss
                + mmd_loss * w_prior
                + prediction_loss * self.w_pred_loss
            )

            self.reconstruction_loss = reconstruction_loss
            self.mmd_loss = mmd_loss
            self.prediction_loss = prediction_loss

            if not validation:
                loss.backward()
                self.optimizer.step()

            epochloss_lst.append(loss.item())

            if (iter + 1) % self.print_every_n_batches == 0:
                if validation:
                    print(
                        f"Epoch {(self.epochs+1):>3d}\tIter {(iter+1):>4d}\tMean Validation Loss:{loss.item():<.7f}\nMean Rec Loss:{reconstruction_loss.item():<.7f}\nMMD Loss:{mmd_loss.item()*w_prior:<.7f}\nMean Pred Loss:{prediction_loss*self.w_pred_loss:<.7f}\n"
                    )
                else:
                    print(
                        f"Epoch {(self.epochs+1):>3d}\tIter {(iter+1):>4d}\tMean Training Loss:{loss.item():<.7f}\nMean Rec Loss:{reconstruction_loss.item():<.7f}\nMMD Loss:{mmd_loss.item()*w_prior:<.7f}\nMean Pred Loss:{prediction_loss*self.w_pred_loss:<.7f}\n"
                    )

        if (self.epochs + 1) % self.print_every_n_epochs == 0:
            if validation:
                print(
                    f"\nEpoch {(self.epochs+1):>3d}\tMean Validation Loss:{sum(epochloss_lst)/len(epochloss_lst):<.7f}\n"
                )
            else:
                print(
                    f"\nEpoch {(self.epochs+1):>3d}\tMean Training Loss:{sum(epochloss_lst)/len(epochloss_lst):<.7f}\n"
                )

            if self.print_topics:
                print(
                    "\n".join(
                        [
                            "{}: {}".format(k, str(v))
                            for k, v in self.get_topic_words().items()
                        ]
                    )
                )
                print("\n")

            if self.content_covariate_size != 0 and self.print_content_covariates:
                print(
                    "\n".join(
                        [
                            "{}: {}".format(k, str(v))
                            for k, v in self.get_covariate_words().items()
                        ]
                    )
                )
                print("\n")

        return sum(epochloss_lst)

    def get_doc_topic_distribution(
        self, dataset, to_simplex=True, num_workers=None, to_numpy=True
    ):
        """
        Get the topic distribution of each document in the corpus.

        Args:
            dataset: a GTMCorpus object
            to_simplex: whether to map the topic distribution to the simplex. If False, the topic distribution is returned in the logit space.
            num_workers: number of workers for the data loaders.
            to_numpy: whether to return the topic distribution as a numpy array.
        """

        if num_workers is None: 
            num_workers = self.num_workers

        with torch.no_grad():
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            final_thetas = []
            for data in data_loader:
                for key, value in data.items():
                    data[key] = value.to(self.device)
                bows = data.get("M_bow", None)
                bows = bows.reshape(bows.shape[0], -1)
                embeddings = data.get("M_embeddings", None)
                prevalence_covariates = data.get("M_prevalence_covariates", None)
                content_covariates = data.get("M_content_covariates", None)
                if self.encoder_input == "bow":
                    x_input = bows
                elif self.encoder_input == "embeddings":
                    x_input = embeddings
                if self.encoder_include_prevalence_covariates:
                    _, thetas = self.AutoEncoder(
                        x_input,
                        prevalence_covariates,
                        content_covariates,
                        to_simplex,
                    )
                else:
                    prevalence_covariates_bis = None
                    _, thetas = self.AutoEncoder(
                        x_input,
                        prevalence_covariates_bis,
                        content_covariates,
                        to_simplex,
                    )
                final_thetas.append(thetas)
            if to_numpy:
                final_thetas = [tensor.cpu().numpy() for tensor in final_thetas]
                final_thetas = np.concatenate(final_thetas, axis=0)
            else:
                final_thetas = torch.cat(final_thetas, dim=0)

        return final_thetas

    def get_predictions(self, dataset, to_simplex=True, num_workers=None, to_numpy=True):
        """
        Predict the labels of the documents in the corpus based on topic proportions.

        Args:
            dataset: a GTMCorpus object
            to_simplex: whether to map the topic distribution to the simplex. If False, the topic distribution is returned in the logit space.
            num_workers: number of workers for the data loaders.
            to_numpy: whether to return the predictions as a numpy array.
        """

        if num_workers is None: 
            num_workers = self.num_workers

        with torch.no_grad():
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            final_predictions = []
            for data in data_loader:
                for key, value in data.items():
                    data[key] = value.to(self.device)
                bows = data.get("M_bow", None)
                bows = bows.reshape(bows.shape[0], -1)
                embeddings = data.get("M_embeddings", None)
                prevalence_covariates = data.get("M_prevalence_covariates", None)
                content_covariates = data.get("M_content_covariates", None)
                prediction_covariates = data.get("M_prediction", None)
                if self.encoder_input == "bow":
                    x_input = bows
                elif self.encoder_input == "embeddings":
                    x_input = embeddings
                if self.encoder_include_prevalence_covariates:
                    _, thetas = self.AutoEncoder(
                        x_input,
                        prevalence_covariates,
                        content_covariates,
                        to_simplex,
                    )
                else:
                    prevalence_covariates_bis = None
                    _, thetas = self.AutoEncoder(
                        x_input,
                        prevalence_covariates_bis,
                        content_covariates,
                        to_simplex,
                    )
                predictions = self.predictor(thetas, prediction_covariates)
                if self.predictor_type == "classifier":
                    predictions = torch.softmax(predictions, dim=1)
                final_predictions.append(predictions)
            if to_numpy:
                final_predictions = [
                    tensor.cpu().numpy() for tensor in final_predictions
                ]
                final_predictions = np.concatenate(final_predictions, axis=0)
            else:
                final_predictions = torch.cat(final_predictions, dim=0)

        return final_predictions

    def get_topic_words(self, l_content_covariates=[], topK=8):
        """
        Get the top words per topic, potentially influenced by content covariates.

        Args:
            l_content_covariates: list with the names of the content covariates to influence the topic-word distribution.
            topK: number of top words to return per topic.
        """
        self.AutoEncoder.eval()
        with torch.no_grad():
            topic_words = {}
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(
                self.device
            )
            for k in l_content_covariates:
                idx = [i for i, l in enumerate(self.content_colnames) if l == k][0]
                idxes[:, (self.n_topics + idx)] += 1
            word_dist = self.AutoEncoder.decode(idxes)
            word_dist = F.softmax(word_dist, dim=1)
            _, indices = torch.topk(word_dist, topK, dim=1)
            indices = indices.cpu().tolist()
            for topic_id in range(self.n_topics):
                topic_words["Topic_{}".format(topic_id)] = [
                    self.id2token[idx] for idx in indices[topic_id]
                ]

        return topic_words

    def get_covariate_words(self, topK=8):
        """
        Get the top words associated to a specific content covariate.

        Args:
            topK: number of top words to return per content covariate.
        """

        self.AutoEncoder.eval()
        with torch.no_grad():
            covariate_words = {}
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(
                self.device
            )
            word_dist = self.AutoEncoder.decode(idxes)
            word_dist = F.softmax(word_dist, dim=1)
            vals, indices = torch.topk(word_dist, topK, dim=1)
            vals = vals.cpu().tolist()
            indices = indices.cpu().tolist()
            for i in range(self.n_topics + self.content_covariate_size):
                if i >= self.n_topics:
                    covariate_words[
                        "{}".format(self.content_colnames[i - self.n_topics])
                    ] = [self.id2token[idx] for idx in indices[i]]
        return covariate_words

    def get_topic_word_distribution(self, l_content_covariates=[], to_numpy=True):
        """
        Get the topic-word distribution of each topic, potentially influenced by covariates.

        Args:
            l_content_covariates: list with the names of the content covariates to influence the topic-word distribution.
            to_numpy: whether to return the topic-word distribution as a numpy array.
        """
        self.AutoEncoder.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(
                self.device
            )
            for k in l_content_covariates:
                idx = [i for i, l in enumerate(self.content_colnames) if l == k][0]
                idxes[:, (self.n_topics + idx)] += 1
            topic_word_distribution = self.AutoEncoder.decode(idxes)
            topic_word_distribution = F.softmax(topic_word_distribution, dim=1)
        if to_numpy:
            return topic_word_distribution.cpu().detach().numpy()[0 : self.n_topics, :]
        else:
            return topic_word_distribution[0 : self.n_topics, :]

    def get_covariate_word_distribution(self, to_numpy=True):
        """
        Get the covariate-word distribution of each topic.

        Args:
            to_numpy: whether to return the covariate-word distribution as a numpy array.
        """
        self.AutoEncoder.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(
                self.device
            )
            topic_word_distribution = self.AutoEncoder.decode(idxes)
            topic_word_distribution = F.softmax(topic_word_distribution, dim=1)
        if to_numpy:
            return topic_word_distribution.cpu().detach().numpy()[self.n_topics :, :]
        else:
            return topic_word_distribution[self.n_topics :, :]

    def get_top_docs(self, dataset, topic_id=None, return_df=False, topK=1):
        """
        Get the most representative documents per topic.

        Args:
            dataset: a GTMCorpus object
            topic_id: the topic to retrieve the top documents from. If None, the top documents for all topics are returned.
            return_df: whether to return the top documents as a DataFrame.
            topK: number of top documents to return per topic. Otherwise, the documents are printed.
        """
        doc_topic_distribution = self.get_doc_topic_distribution(dataset)
        top_k_indices_df = pd.DataFrame(
            {
                f"Topic_{col}": top_k_indices_column(
                    doc_topic_distribution[:, col], topK
                )
                for col in range(doc_topic_distribution.shape[1])
            }
        )
        if return_df is False:
            if topic_id is None:
                for topic_id in range(self.n_topics):
                    for i in top_k_indices_df["Topic_{}".format(topic_id)]:
                        print(
                            "Topic: {} | Document index: {} | Topic share: {}".format(
                                topic_id, i, doc_topic_distribution[i, topic_id]
                            )
                        )
                        print(dataset.df["doc"].iloc[i])
                        print("\n")
            else:
                for i in top_k_indices_df["Topic_{}".format(topic_id)]:
                    print(
                        "Topic: {} | Document index: {} | Topic share: {}".format(
                            topic_id, i, doc_topic_distribution[i, topic_id]
                        )
                    )
                    print(dataset.df["doc"].iloc[i])
                    print("\n")
        else:
            l = []
            for topic_id in range(self.n_topics):
                for i in top_k_indices_df["Topic_{}".format(topic_id)]:
                    d = {}
                    d["topic_id"] = topic_id
                    d["doc_id"] = i
                    d["topic_share"] = doc_topic_distribution[i, topic_id]
                    d["doc"] = dataset.df["doc"].iloc[i]
                    l.append(d)
            df = pd.DataFrame.from_records(l)
            if topic_id is not None:
                df = df[df["topic_id"] == topic_id]
                df = df.reset_index(drop=True)
            return df

    def plot_topic_word_distribution(
        self,
        topic_id,
        content_covariates=[],
        topK=100,
        plot_type="wordcloud",
        output_path=None,
        wordcloud_args={"background_color": "white"},
        plt_barh_args={"color": "grey"},
        plt_savefig_args={"dpi": 300},
    ):
        """
        Returns a wordcloud/barplot representation per topic.

        Args:
            topic_id: the topic to visualize.
            content_covariates: list with the names of the content covariates to influence the topic-word distribution.
            topK: number of top words to return per topic.
            plot_type: either 'wordcloud' or 'barplot'.
            output_path: path to save the plot.
            wordcloud_args: dictionary with the parameters for the wordcloud plot.
            plt_barh_args: dictionary with the parameters for the barplot plot.
            plt_savefig_args: dictionary with the parameters for the savefig function.
        """

        topic_word_distribution = self.get_topic_word_distribution(
            content_covariates, to_numpy=False
        )
        vals, indices = torch.topk(topic_word_distribution, topK, dim=1)
        vals = vals.cpu().tolist()
        indices = indices.cpu().tolist()
        topic_words = [self.id2token[idx] for idx in indices[topic_id]]
        values = vals[topic_id]
        d = {}
        for i, w in enumerate(topic_words):
            d[w] = values[i]

        if plot_type == "wordcloud":
            wordcloud = WordCloud(**wordcloud_args).generate_from_frequencies(d)
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
        else:
            sorted_items = sorted(d.items(), key=lambda x: x[1])
            words = [item[0] for item in sorted_items]
            values = [item[1] * 100 for item in sorted_items]
            plt.figure(figsize=(8, len(words) // 2))
            plt.barh(words, values, **plt_barh_args)
            plt.xlabel("Probability")
            plt.ylabel("Words")
            plt.title("Words for {}".format(self.topic_labels[topic_id]))
            plt.show()

        if output_path is not None:
            plt.savefig(output_path, **plt_savefig_args)

    def visualize_docs(
        self,
        dataset,
        dimension_reduction="tsne",
        dimension_reduction_args={"random_state": 42},
        update_layout_args=dict(
            autosize=True,
            width=None,
            height=None,
            plot_bgcolor="white",
            paper_bgcolor="white",
        ),
        display=True,
        output_path=None,
    ):
        """
        Visualize the documents in the corpus based on their topic distribution.

        Args:
            dataset: a GTMCorpus object
            dimension_reduction: dimensionality reduction technique. Either 'umap', 'tsne' or 'pca'.
            dimension_reduction_args: dictionary with the parameters for the dimensionality reduction technique.
            update_layout_args: dictionary with the parameters for the layout of the plot.
            display: whether to display the plot.
            output_path: path to save the plot.
        """

        matrix = self.get_doc_topic_distribution(dataset)
        most_prevalent_topics = np.argmax(matrix, axis=1)
        most_prevalent_topic_share = np.max(matrix, axis=1)

        if dimension_reduction == "umap":
            ModelLowDim = UMAP(n_components=2, **dimension_reduction_args)
        if dimension_reduction == "tsne":
            ModelLowDim = TSNE(n_components=2, **dimension_reduction_args)
        else:
            ModelLowDim = PCA(n_components=2, **dimension_reduction_args)

        EmbeddingsLowDim = ModelLowDim.fit_transform(matrix)

        labels = list(dataset.df["doc_clean"])

        deciles = np.percentile(most_prevalent_topic_share, np.arange(0, 100, 10))
        marker_sizes = np.zeros_like(most_prevalent_topic_share)
        for i in range(1, 10):
            marker_sizes[
                (most_prevalent_topic_share > deciles[i - 1])
                & (most_prevalent_topic_share <= deciles[i])
            ] = i

        trace = go.Scatter(
            x=EmbeddingsLowDim[:, 0],
            y=EmbeddingsLowDim[:, 1],
            mode="markers",
            text=labels,
            hoverinfo="text",
            marker=dict(
                size=marker_sizes,
                color=most_prevalent_topics,
                colorscale="Plasma",
                opacity=0.5,
            ),
        )
        annotations = []
        for i, topic_name in enumerate(self.topic_labels):
            annotations.append(
                dict(
                    x=EmbeddingsLowDim[most_prevalent_topics == i, 0].mean(),
                    y=EmbeddingsLowDim[most_prevalent_topics == i, 1].mean(),
                    xref="x",
                    yref="y",
                    text='<b> <span style="font-size: 16px;">'
                    + topic_name
                    + "</span> </b>",
                    showarrow=False,
                    ax=0,
                    ay=0,
                )
            )
        layout = go.Layout(hovermode="closest", annotations=annotations)
        fig = go.Figure(data=[trace], layout=layout)
        fig.update_layout(**update_layout_args)
        if display:
            fig.show(config=dict(editable=True))
        if output_path is not None:
            fig.write_html(output_path, config=dict(editable=True))

    def visualize_words(
        self,
        dimension_reduction="tsne",
        dimension_reduction_args={"random_state": 42},
        update_layout_args=dict(
            autosize=True,
            width=None,
            height=None,
            plot_bgcolor="white",
            paper_bgcolor="white",
        ),
        display=True,
        output_path=None,
    ):
        """
        Visualize the words in the corpus based on their topic distribution.

        Args:
            dimension_reduction: dimensionality reduction technique. Either 'umap', 'tsne' or 'pca'.
            dimension_reduction_args: dictionary with the parameters for the dimensionality reduction technique.
            update_layout_args: dictionary with the parameters for the layout of the plot.
            display: whether to display the plot.
            output_path: path to save the plot.
        """

        matrix = self.get_topic_word_distribution().T
        most_prevalent_topics = np.argmax(matrix, axis=1)
        most_prevalent_topic_share = np.max(matrix, axis=1)

        if dimension_reduction == "umap":
            ModelLowDim = UMAP(n_components=2, **dimension_reduction_args)
        if dimension_reduction == "tsne":
            ModelLowDim = TSNE(n_components=2, **dimension_reduction_args)
        else:
            ModelLowDim = PCA(n_components=2, **dimension_reduction_args)

        EmbeddingsLowDim = ModelLowDim.fit_transform(matrix)

        labels = list(self.id2token.values())

        deciles = np.percentile(most_prevalent_topic_share, np.arange(0, 100, 10))
        marker_sizes = np.zeros_like(most_prevalent_topic_share)
        for i in range(1, 10):
            marker_sizes[
                (most_prevalent_topic_share > deciles[i - 1])
                & (most_prevalent_topic_share <= deciles[i])
            ] = i

        trace = go.Scatter(
            x=EmbeddingsLowDim[:, 0],
            y=EmbeddingsLowDim[:, 1],
            mode="markers",
            text=labels,
            hoverinfo="text",
            marker=dict(
                size=marker_sizes,
                color=most_prevalent_topics,
                colorscale="Plasma",
                opacity=0.5,
            ),
        )
        annotations = []
        top_words = [v for k, v in self.get_topic_words(topK=1).items()]
        for l in top_words:
            for word in l:
                annotations.append(
                    dict(
                        x=EmbeddingsLowDim[labels.index(word), 0],
                        y=EmbeddingsLowDim[labels.index(word), 1],
                        xref="x",
                        yref="y",
                        text='<b> <span style="font-size: 16px;">'
                        + word
                        + "</b> </span>",
                        showarrow=False,
                        ax=0,
                        ay=0,
                    )
                )
        layout = go.Layout(hovermode="closest", annotations=annotations)
        fig = go.Figure(data=[trace], layout=layout)
        fig.update_layout(**update_layout_args)
        if display:
            fig.show(config=dict(editable=True))
        if output_path is not None:
            fig.write_html(output_path, config=dict(editable=True))

    def visualize_topics(
        self,
        dataset,
        dimension_reduction_args={},
        update_layout_args=dict(
            autosize=True,
            width=None,
            height=None,
            plot_bgcolor="white",
            paper_bgcolor="white",
        ),
        display=True,
        output_path=None,
    ):
        """
        Visualize the topics in the corpus based on their topic distribution.

        Args:
            dataset: a GTMCorpus object
            dimension_reduction_args: dictionary with the parameters for the dimensionality reduction technique.
            update_layout_args: dictionary with the parameters for the layout of the plot.
            display: whether to display the plot.
            output_path: path to save the plot.
        """

        matrix = self.get_topic_word_distribution()
        doc_topic_dist = self.get_doc_topic_distribution(dataset)
        df = pd.DataFrame(doc_topic_dist)
        marker_sizes = np.array(df.mean()) * 1000
        ModelLowDim = PCA(n_components=2, **dimension_reduction_args)
        EmbeddingsLowDim = ModelLowDim.fit_transform(matrix)
        labels = [v for k, v in self.get_topic_words().items()]

        trace = go.Scatter(
            x=EmbeddingsLowDim[:, 0],
            y=EmbeddingsLowDim[:, 1],
            mode="markers",
            text=labels,
            hoverinfo="text",
            marker=dict(size=marker_sizes),
        )
        annotations = []
        for i, topic_name in enumerate(self.topic_labels):
            annotations.append(
                dict(
                    x=EmbeddingsLowDim[i, 0],
                    y=EmbeddingsLowDim[i, 1],
                    xref="x",
                    yref="y",
                    text='<b> <span style="font-size: 16px;">'
                    + topic_name
                    + "</span> </b>",
                    showarrow=False,
                    ax=0,
                    ay=0,
                )
            )
        layout = go.Layout(hovermode="closest", annotations=annotations)
        fig = go.Figure(data=[trace], layout=layout)
        fig.update_layout(**update_layout_args)
        if display:
            fig.show(config=dict(editable=True))
        if output_path is not None:
            fig.write_html(output_path, config=dict(editable=True))

    def estimate_effect(self, dataset, n_samples=20, topic_ids=None, progress_bar=True):
        """
        GLM estimates and associated standard errors of the doc-topic prior conditional on the prevalence covariates.

        Uncertainty is computed using the method of composition.
        Technically, this means we draw a set of topic proportions from the variational posterior, 
        run a regression topic_proportion ~ covariates, then repeat for n_samples.  
        Quantities of interest are the mean and the standard deviation of regression coefficients across samples.

        /!\ May take quite some time to run. /!\

        References:
        - Roberts, M. E., Stewart, B. M., & Airoldi, E. M. (2016). A model of text for experimentation in the social sciences. Journal of the American Statistical Association, 111(515), 988-1003.
        """

        X = dataset.M_prevalence_covariates

        if topic_ids is None:
            iterator = range(self.n_topics)
        else:
            iterator = topic_ids

        if progress_bar:
            samples_iterator = tqdm(range(n_samples))
        else:
            samples_iterator = range(n_samples)

        dict_of_params = {"Topic_{}".format(k): [] for k in range(self.n_topics)}
        OLS_model = LinearRegression(fit_intercept=False)
        for i in samples_iterator:
            Y = self.prior.sample(X.shape[0], X).cpu().numpy()
            for k in iterator:
                OLS_model.fit(
                    X[
                        :,
                    ],
                    Y[:, k] * 100,
                )
                dict_of_params["Topic_{}".format(k)].append(np.array([OLS_model.coef_]))

        records_for_df = []
        for k in iterator:
            d = {}
            d["topic"] = k
            a = np.concatenate(dict_of_params["Topic_{}".format(k)])
            mean = np.mean(a, axis=0)
            sd = np.std(a, axis=0)
            for i, cov in enumerate(dataset.prevalence_colnames):
                d = {}
                d["topic"] = k
                d["covariate"] = cov
                d["mean"] = mean[i]
                d["sd"] = sd[i]
                records_for_df.append(d)

        df = pd.DataFrame.from_records(records_for_df)

        return df

    def get_ldavis_data_format(self, dataset):
        """
        Returns a data format that can be used in input to pyldavis to interpret the topics.
        """
        term_frequency = np.ravel(dataset.M_bow.sum(axis=0))
        doc_lengths = np.ravel(dataset.M_bow.sum(axis=1))
        vocab = self.id2token
        term_topic = self.get_topic_word_distribution()
        doc_topic_distribution = self.get_doc_topic_distribution(dataset)

        data = {
            "topic_term_dists": term_topic,
            "doc_topic_dists": doc_topic_distribution,
            "doc_lengths": doc_lengths,
            "vocab": vocab,
            "term_frequency": term_frequency,
        }

        return data

    def save_model(self, save_name):
        autoencoder_state_dict = self.AutoEncoder.state_dict()
        if self.labels_size != 0:
            predictor_state_dict = self.predictor.state_dict()
        else:
            predictor_state_dict = None
        optimizer_state_dict = self.optimizer.state_dict()

        all_vars = vars(self)

        checkpoint = {}
        for key, value in all_vars.items():
            if key not in ["AutoEncoder", "predictor", "optimizer"]:
                checkpoint[key] = value

        checkpoint["AutoEncoder"] = autoencoder_state_dict
        if self.labels_size != 0:
            checkpoint["predictor"] = predictor_state_dict
        checkpoint["optimizer"] = optimizer_state_dict

        torch.save(checkpoint, save_name)

    def load_model(self, ckpt):
        """
        Helper function to load a GTM model.
        """
        ckpt = torch.load(ckpt)
        for key, value in ckpt.items():
            if key not in ["AutoEncoder", "predictor", "optimizer"]:
                setattr(self, key, value)

        if not hasattr(self, "AutoEncoder"):
            if self.encoder_include_prevalence_covariates:
                encoder_dims = [self.input_size + self.prevalence_covariate_size]
            else:
                encoder_dims = [self.input_size]
            encoder_dims.extend(self.encoder_hidden_layers)
            encoder_dims.extend([self.n_topics])

            decoder_dims = [self.n_topics + self.content_covariate_size]
            decoder_dims.extend(self.decoder_hidden_layers)
            decoder_dims.extend([self.bow_size])

            self.AutoEncoder = AutoEncoderMLP(
                encoder_dims=encoder_dims,
                encoder_non_linear_activation=self.encoder_non_linear_activation,
                encoder_bias=self.encoder_bias,
                decoder_dims=decoder_dims,
                decoder_non_linear_activation=self.decoder_non_linear_activation,
                decoder_bias=self.decoder_bias,
                dropout=self.dropout,
            ).to(self.device)

        self.AutoEncoder.load_state_dict(ckpt["AutoEncoder"])

        if self.labels_size != 0:
            if not hasattr(self, "predictor"):
                self.predictor = Predictor(
                    predictor_dims=[self.n_topics + self.prediction_covariate_size]
                    + self.predictor_hidden_layers
                    + [self.labels_size],
                    predictor_non_linear_activation=self.predictor_non_linear_activation,
                    predictor_bias=self.predictor_bias,
                    dropout=self.dropout,
                ).to(self.device)
            self.predictor.load_state_dict(ckpt["predictor"])

        if not hasattr(self, "optimizer"):
            if self.labels_size != 0:
                self.optimizer = torch.optim.Adam(
                    list(self.AutoEncoder.parameters())
                    + list(self.predictor.parameters()),
                    lr=self.learning_rate,
                )
            else:
                self.optimizer = torch.optim.Adam(
                    self.AutoEncoder.parameters(), lr=self.learning_rate
                )
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def to(self, device):
        """
        Move the model to a different device.
        """
        self.AutoEncoder.to(device)
        self.prior.to(device)
        if self.labels_size != 0:
            self.predictor.to(device)
        self.device = device
