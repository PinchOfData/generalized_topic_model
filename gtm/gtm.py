#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from autoencoders import EncoderMLP, DecoderMLP, MultiModalEncoderMoE
from predictors import Predictor
from priors import DirichletPrior, LogisticNormalPrior
from utils import compute_mmd_loss, top_k_indices_column
import os
import torch
import numpy as np
import torch.multiprocessing as mp
from typing import Optional, List, Dict, Union
from corpus import GTMCorpus
from collections import OrderedDict
import statsmodels.api as sm

def parse_modality_view(key: str):
    if '_' not in key:
        raise ValueError(f"Expected 'modality.view', got '{key}'")
    return key.split('_', 1)

class GTM:
    def __init__(
        self,
        train_data: Optional[GTMCorpus] = None,
        test_data: Optional[GTMCorpus] = None,
        n_topics: int = 20,
        ae_type: str = "wae",
        doc_topic_prior: str = "dirichlet",
        update_prior: bool = False,
        alpha: float = 0.1,
        prevalence_model_type: str = "RidgeCV",
        prevalence_model_args: Dict = {},
        tol: float = 0.001,
        encoder_input: Union[str, List[str]] = "default_bow",
        decoder_input: Union[str, List[str]] = "default_bow",
        encoder_hidden_layers: List[int] = [256],
        encoder_non_linear_activation: str = "relu",
        encoder_bias: bool = True,
        encoder_gating: bool = False,
        decoder_hidden_layers: List[int] = [],
        decoder_non_linear_activation: Optional[str] = None,
        decoder_bias: bool = False,
        predictor_type: Optional[str] = None,
        predictor_hidden_layers: List[int] = [],
        predictor_non_linear_activation: str = "relu",
        predictor_bias: bool = True,
        initialization: Optional[bool] = True,
        num_epochs: int = 1000,
        num_workers: Optional[int] = 4,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
        regularization: float = 0,
        print_every_n_epochs: int = 1,
        print_every_n_batches: int = 10000,
        print_topics: bool = True,
        print_content_covariates: bool = True,
        log_every_n_epochs: int = 10000,
        patience: int = 1,
        delta: float = 0.0,
        w_prior: Union[int, None] = 1,
        w_pred_loss: int = 1,
        kl_annealing_start: int = 0,
        kl_annealing_end: int = 100,
        kl_annealing_max_beta: float = 1.0,
        ckpt_folder: str = "../ckpt",
        ckpt: Optional[str] = None,
        device: Optional[torch.device] = None,
        seed: Optional[int] = 42,
    ):
        """
        Args:
            train_data: a GTMCorpus object
            test_data: a GTMCorpus object
            n_topics: number of topics
            ae_type: type of autoencoder. Either 'wae' (Wasserstein Autoencoder) or 'vae' (Variational Autoencoder).
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
            kl_annealing_start: epoch at which to start the KL annealing.
            kl_annealing_end: epoch at which to end the KL annealing.
            kl_annealing_max_beta: maximum value of the KL annealing beta.
            ckpt_folder: folder to save the checkpoints.
            ckpt: checkpoint to load the model from.
            device: device to use for training.
            seed: random seed.
        """        

        if isinstance(encoder_input, str):
            encoder_input = [encoder_input]
        if isinstance(decoder_input, str):
            decoder_input = [decoder_input]

        if ckpt:
            self.load_model(ckpt)
            return

        self.n_topics = n_topics
        self.ae_type = ae_type
        self.topic_labels = [f"Topic_{i}" for i in range(n_topics)]
        self.doc_topic_prior = doc_topic_prior
        self.update_prior = update_prior
        self.alpha = alpha
        self.prevalence_model_type = prevalence_model_type
        self.prevalence_model_args = prevalence_model_args
        self.tol = tol
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input

        self.encoder_hidden_layers = encoder_hidden_layers
        self.encoder_non_linear_activation = encoder_non_linear_activation
        self.encoder_bias = encoder_bias
        self.encoder_gating = encoder_gating
        self.decoder_hidden_layers = decoder_hidden_layers
        self.decoder_non_linear_activation = decoder_non_linear_activation
        self.decoder_bias = decoder_bias
        self.predictor_type = predictor_type
        self.predictor_hidden_layers = predictor_hidden_layers
        self.predictor_non_linear_activation = predictor_non_linear_activation
        self.predictor_bias = predictor_bias

        self.initialization = initialization
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cpu")
        )
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
        self.dropout = dropout
        self.regularization = regularization

        self.print_every_n_epochs = print_every_n_epochs
        self.print_every_n_batches = print_every_n_batches
        self.print_topics = print_topics
        self.print_content_covariates = print_content_covariates
        self.log_every_n_epochs = log_every_n_epochs
        self.patience = patience
        self.delta = delta
        self.w_prior = w_prior
        self.kl_annealing_start = kl_annealing_start
        self.kl_annealing_end = kl_annealing_end
        self.kl_annealing_max_beta = kl_annealing_max_beta
        self.w_pred_loss = w_pred_loss
        self.ckpt_folder = ckpt_folder

        if self.ae_type == "vae" and self.doc_topic_prior == "dirichlet":
            raise ValueError("VAE cannot be used with a Dirichlet prior. Use 'logistic_normal' instead.")

        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)

        if seed is not None:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            np.random.seed(seed)

        self.prevalence_covariate_size = (
            train_data.M_prevalence_covariates.shape[1] if train_data.prevalence else 0
        )
        self.content_covariate_size = (
            train_data.M_content_covariates.shape[1] if train_data.content else 0
        )
        self.prediction_covariate_size = (
            train_data.M_prediction.shape[1] if train_data.prediction else 0
        )
        self.labels_size = (
            train_data.M_labels.shape[1] if train_data.labels else 0
        )

        self.content_colnames = train_data.content_colnames or []
        self.id2token = train_data.id2token

        # Predictor output size
        if predictor_type == "classifier" and train_data.labels is not None:
            n_labels = len(np.unique(train_data.M_labels))
        else:
            n_labels = 1

        # ENCODER
        encoders = OrderedDict()
        for key in encoder_input:
            modality, view = parse_modality_view(key)
            input_dim = train_data.processed_modalities[modality][view]["matrix"].shape[1]
            enc_dims = [input_dim + self.prevalence_covariate_size] 
            enc_dims += encoder_hidden_layers 
            enc_dims += [n_topics * 2 if ae_type == "vae" else n_topics]
            encoders[key] = EncoderMLP(
                encoder_dims=enc_dims,
                encoder_non_linear_activation=encoder_non_linear_activation,
                encoder_bias=encoder_bias,
                dropout=dropout,
            )
        self.encoder = MultiModalEncoderMoE(
            encoders=encoders,
            topic_dim=n_topics,
            gating=self.encoder_gating,
            ae_type=self.ae_type,
        ).to(self.device)

        # DECODERS
        self.decoders = nn.ModuleDict()
        for key in decoder_input:
            modality, view = parse_modality_view(key)
            output_dim = train_data.processed_modalities[modality][view]["matrix"].shape[1]
            dec_dims = [n_topics + self.content_covariate_size] + decoder_hidden_layers + [output_dim]
            self.decoders[key] = DecoderMLP(
                decoder_dims=dec_dims,
                decoder_non_linear_activation=decoder_non_linear_activation,
                decoder_bias=decoder_bias,
                dropout=dropout,
            ).to(self.device)

        # PRIOR
        if doc_topic_prior == "dirichlet":
            self.prior = DirichletPrior(
                update_prior,
                self.prevalence_covariate_size,
                self.n_topics,
                alpha,
                prevalence_model_args,
                tol,
                device=self.device,
            )
        elif doc_topic_prior == "logistic_normal":
            self.prior = LogisticNormalPrior(
                self.prevalence_covariate_size,
                n_topics,
                prevalence_model_type,
                prevalence_model_args,
                device=self.device,
            )

        # PREDICTOR
        if self.labels_size != 0:
            predictor_dims = [n_topics + self.prediction_covariate_size] + predictor_hidden_layers + [n_labels]
            self.predictor = Predictor(
                predictor_dims=predictor_dims,
                predictor_non_linear_activation=predictor_non_linear_activation,
                predictor_bias=predictor_bias,
                dropout=dropout,
            ).to(self.device)

        # OPTIMIZER
        all_params = list(self.encoder.parameters()) + list(self.decoders.parameters())
        if self.labels_size != 0:
            all_params += list(self.predictor.parameters())
        self.optimizer = torch.optim.Adam(
            all_params, lr=learning_rate, betas=(0.99, 0.999)
        )

        self.epochs = 0
        self.loss = np.inf
        self.reconstruction_loss = np.inf
        self.mmd_loss = np.inf
        self.prediction_loss = np.inf

        if self.initialization and self.update_prior:
            self.initialize(train_data, test_data)

        self.train(train_data, test_data)

    def initialize(self, train_data, test_data=None):
        """
        Train a rough initial model using Adam optimizer without modifying the learning rate.
        Stops as soon as the validation loss stops improving (patience == 1).
        Saves the best model before the loss stopped improving.
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

        best_loss = np.inf
        counter = 0
        best_model_path = f"{self.ckpt_folder}/best_initial_model.ckpt"

        print('Initializing model...')

        for epoch in range(self.num_epochs):
            training_loss = self.epoch(train_data_loader, validation=False, initialization=True)
            
            if test_data is not None:
                validation_loss = self.epoch(test_data_loader, validation=True, initialization=True)
                current_loss = validation_loss
            else:
                current_loss = training_loss

            loss_improved = current_loss < best_loss

            if loss_improved:
                best_loss = current_loss
                counter = 0  
                self.save_model(best_model_path)
            else:
                counter += 1

            if counter >= 1:
                print(f"Initialization completed in {epoch+1} epochs.")
                break

        self.load_model(best_model_path)

        if self.update_prior:
            if self.doc_topic_prior == "dirichlet":
                posterior_theta = self.get_doc_topic_distribution(train_data, to_numpy=True)
                self.prior.update_parameters(
                    posterior_theta, train_data.M_prevalence_covariates
                )
            else:
                posterior_theta = self.get_doc_topic_distribution(
                    train_data, to_simplex=False, to_numpy=True
                )
                self.prior.update_parameters(
                    posterior_theta, train_data.M_prevalence_covariates
                )             

    def train(self, train_data, test_data=None):
        """
        Train the model.
        """

        current_lr = self.learning_rate

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
        self.save_model("{}/best_model.ckpt".format(self.ckpt_folder))

        if self.epochs == 0:
            best_loss = np.inf
            best_epoch = -1

        else:
            best_loss = self.loss
            best_epoch = self.epochs

        for epoch in range(self.epochs, self.num_epochs):

            training_loss = self.epoch(train_data_loader, validation=False)

            if test_data is not None:
                validation_loss = self.epoch(test_data_loader, validation=True)

            if (epoch + 1) % self.log_every_n_epochs == 0:
                save_name = f'{self.ckpt_folder}/GTM_K{self.n_topics}_{self.doc_topic_prior}_{self.predictor_type}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_{self.epochs+1}.ckpt'
                self.save_model(save_name)

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

            if counter >= self.patience or (epoch + 1) == self.num_epochs:

                ckpt = "{}/best_model.ckpt".format(self.ckpt_folder)
                self.load_model(ckpt)

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr*0.1
                    current_lr = param_group['lr']

                if current_lr < 1e-3:
                    print(
                        "\nStopping at Epoch {}. Reverting to Epoch {}".format(
                            self.epochs + 1, best_epoch + 1
                        )
                    )
                    ckpt = "{}/best_model.ckpt".format(self.ckpt_folder)
                    self.load_model(ckpt)
                    break

            self.epochs += 1

    def epoch(self, data_loader, validation=False, initialization=False, num_samples=1):
        """
        Train the model for one epoch.
        """
        if validation:
            self.encoder.eval()
            self.decoders.eval()
            if self.labels_size != 0:
                self.predictor.eval()
        else:
            self.encoder.train()
            self.decoders.train()
            if self.labels_size != 0:
                self.predictor.train()

        epochloss_lst = []
        all_topics = []
        all_prevalence_covariates = []

        with torch.no_grad() if validation else torch.enable_grad():
            for iter, data in enumerate(data_loader):
                if not validation:
                    self.optimizer.zero_grad()

                # Move all tensors to device
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.to(self.device)

                prevalence_covariates = data.get("M_prevalence_covariates", None)
                content_covariates = data.get("M_content_covariates", None)
                prediction_covariates = data.get("M_prediction", None)
                target_labels = data.get("M_labels", None)

                # -------------------- ENCODER INPUT --------------------
                modality_inputs = {}
                for key in self.encoder_input:
                    mod, view = parse_modality_view(key)
                    x = data["modalities"][mod][view].to(self.device)

                    if prevalence_covariates is not None:
                        x = torch.cat([x, prevalence_covariates], dim=1)

                    modality_inputs[key] = x

                theta_q, z, mu_logvar = self.encoder(modality_inputs)
                #if self.ae_type == "vae" and not validation:
                #    with torch.no_grad():
                #       mus = torch.cat([mu for (mu, _) in mu_logvar], dim=0)
                #       print("Mean μ:", mus.mean().item(), "Std μ:", mus.std().item())

                # -------------------- DECODERS --------------------
                reconstruction_loss = 0.0
                for key, decoder in self.decoders.items():
                    mod, view = parse_modality_view(key)
                    target = data["modalities"][mod][view].to(self.device)
                    theta_input = torch.cat([theta_q, content_covariates], dim=1) if content_covariates is not None else theta_q
                    recon = decoder(theta_input)

                    view_type = data_loader.dataset.processed_modalities[mod][view]["type"]

                    if view_type == "bow":
                        log_probs = F.log_softmax(recon, dim=1)
                        recon_loss = -torch.sum(target * log_probs) / torch.sum(target)

                    elif view_type == "embedding":
                        recon_loss = F.mse_loss(recon, target)

                    else:
                        raise ValueError(f"Unsupported view type '{view_type}' for '{mod}_{view}'")

                    reconstruction_loss += recon_loss

                # -------------------- PRIOR / MMD --------------------
                mmd_loss = 0.0
                for _ in range(num_samples):
                    theta_prior = self.prior.sample(
                        N=theta_q.shape[0],
                        M_prevalence_covariates=prevalence_covariates,
                        epoch=self.epochs,
                        initialization=initialization
                    ).to(self.device)
                    mmd_loss += compute_mmd_loss(theta_q, theta_prior, device=self.device)

                if self.w_prior is None:
                    total_tokens = sum(data["modalities"][mod][view].sum() for mod, view in [parse_modality_view(k)] for k in self.decoder_input)
                    mean_length = total_tokens / theta_q.shape[0]
                    vocab_size = list(data["modalities"][mod][view].shape[1] for mod, view in [parse_modality_view(k)] for k in self.decoder_input)[0]
                    w_prior = mean_length * np.log(vocab_size)
                else:
                    w_prior = self.w_prior

                if self.epochs < self.kl_annealing_start:
                    beta = 0.0
                elif self.epochs > self.kl_annealing_end:
                    beta = self.kl_annealing_max_beta
                else:
                    progress = (self.epochs - self.kl_annealing_start) / (self.kl_annealing_end - self.kl_annealing_start)
                    beta = progress * self.kl_annealing_max_beta

                if self.ae_type == "vae" and self.update_prior:
                    kl_loss = 0.0
                    for (mu_q, logvar_q) in mu_logvar:
                        mu_p, logvar_p = self.prior.get_prior_params(prevalence_covariates)
                        
                        # Diagonal KL
                        kl = 0.5 * torch.sum(
                            logvar_p - logvar_q + 
                            (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp() - 1,
                            dim=1
                        ).mean()
                        kl_loss += kl
                    divergence_loss = beta*kl_loss
                elif self.ae_type == "vae" and not self.update_prior:
                    kl_loss = 0.0
                    for mu, logvar in mu_logvar:
                        kl_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                    divergence_loss = beta*kl_loss
                else:
                    divergence_loss = mmd_loss*w_prior

                # -------------------- PREDICTION --------------------
                if target_labels is not None:
                    predictions = self.predictor(theta_q, prediction_covariates)
                    if self.predictor_type == "classifier":
                        target_labels = target_labels.squeeze().to(torch.int64)
                        prediction_loss = F.cross_entropy(predictions, target_labels)
                    elif self.predictor_type == "regressor":
                        prediction_loss = F.mse_loss(predictions, target_labels)
                else:
                    prediction_loss = 0.0

                # -------------------- L2 Regularization --------------------
                l2_norm = sum(torch.norm(param, p=2) for param in self.encoder.parameters())
                for decoder in self.decoders.values():
                    l2_norm += sum(torch.norm(param, p=2) for param in decoder.parameters())

                # -------------------- TOTAL LOSS --------------------
                loss = (
                    reconstruction_loss
                    + divergence_loss
                    + prediction_loss * self.w_pred_loss
                    + self.regularization * l2_norm
                )

                self.loss = loss
                self.reconstruction_loss = reconstruction_loss
                self.mmd_loss = mmd_loss
                self.prediction_loss = prediction_loss

                if not validation:
                    loss.backward()
                    self.optimizer.step()

                epochloss_lst.append(loss.item())

                if self.update_prior and not validation and not initialization:
                    if self.doc_topic_prior == "logistic_normal":
                        all_topics.append(z.detach().cpu())
                    else:
                        all_topics.append(theta_q.detach().cpu())
                    all_prevalence_covariates.append(prevalence_covariates.detach().cpu())

                if (iter + 1) % self.print_every_n_batches == 0:
                    msg = (
                        f"Epoch {(self.epochs+1):>3d}\tIter {(iter+1):>4d}"
                        f"\tMean {'Validation' if validation else 'Training'} Loss:{loss.item():<.7f}"
                        f"\nRec Loss:{reconstruction_loss.item():<.7f}"
                        f"\nDivergence Loss:{divergence_loss.item():<.7f}"
                        f"\nPred Loss:{prediction_loss * self.w_pred_loss:<.7f}\n"
                    )
                    print(msg)

        # -------------------- END OF EPOCH --------------------
        if (self.epochs + 1) % self.print_every_n_epochs == 0 and initialization == False:
            avg_loss = sum(epochloss_lst) / len(epochloss_lst)
            print(
                f"\nEpoch {(self.epochs+1):>3d}\tMean {'Validation' if validation else 'Training'} Loss:{avg_loss:<.7f}\n"
            )

            if self.print_topics:
                print("\n".join(
                    f"{k}: {v}" for k, v in self.get_topic_words().items()
                ), "\n")

            if self.content_covariate_size != 0 and self.print_content_covariates:
                print("\n".join(
                    f"{k}: {v}" for k, v in self.get_covariate_words().items()
                ), "\n")

        if self.update_prior and not validation and not initialization:
            all_topics = torch.cat(all_topics, dim=0).numpy()
            all_prevalence_covariates = torch.cat(all_prevalence_covariates, dim=0).numpy()
            self.prior.update_parameters(all_topics, all_prevalence_covariates)

        return sum(epochloss_lst)

    def get_doc_topic_distribution(
        self,
        dataset,
        to_simplex: bool = True,
        num_workers: Optional[int] = None,
        to_numpy: bool = True,
        single_modality: Optional[str] = None,
        num_samples: int = 1,
    ):
        """
        Get the topic distribution of each document in the corpus.

        Args:
            dataset: a GTMCorpus object
            to_simplex: whether to map the topic distribution to the simplex. If False, returns latent logits.
            num_workers: number of workers for the data loaders.
            to_numpy: whether to return as a numpy array.
            single_modality: if set, uses only this modality (e.g., "default_bow")
            num_samples: number of samples from the VAE encoder (only used for VAE).
        """
        if num_workers is None:
            num_workers = self.num_workers

        self.encoder.eval()
        final_thetas = []

        with torch.no_grad():
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            for data in data_loader:
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.to(self.device)

                prevalence_covariates = data.get("M_prevalence_covariates", None)

                if single_modality is not None:
                    # Single modality path
                    mod, view = parse_modality_view(single_modality)
                    x = data["modalities"][mod][view].to(self.device)
                    if prevalence_covariates is not None:
                        x = torch.cat([x, prevalence_covariates], dim=1)

                    z = self.encoder.encoders[single_modality](x)

                    if self.ae_type == "vae":
                        mu, logvar = torch.chunk(z, 2, dim=1)
                        thetas = []
                        for _ in range(num_samples):
                            std = torch.exp(0.5 * logvar)
                            eps = torch.randn_like(std)
                            z_sampled = mu + eps * std
                            theta = F.softmax(z_sampled, dim=1) if to_simplex else z_sampled
                            thetas.append(theta)
                        theta_q = torch.stack(thetas, dim=1).mean(dim=1)
                    else:
                        theta_q = F.softmax(z, dim=1) if to_simplex else z

                else:
                    # Multimodal path
                    modality_inputs = {}
                    for key in self.encoder_input:
                        mod, view = parse_modality_view(key)
                        x = data["modalities"][mod][view].to(self.device)
                        if prevalence_covariates is not None:
                            x = torch.cat([x, prevalence_covariates], dim=1)
                        modality_inputs[key] = x

                    if self.ae_type == "vae":
                        thetas = []
                        for _ in range(num_samples):
                            theta_q, z, _ = self.encoder(modality_inputs)
                            theta_q = theta_q if to_simplex else z
                            thetas.append(theta_q)
                        theta_q = torch.stack(thetas, dim=1).mean(dim=1)
                    else:
                        theta_q, z, _ = self.encoder(modality_inputs)
                        theta_q = theta_q if to_simplex else z

                final_thetas.append(theta_q)

            if to_numpy:
                final_thetas = [t.cpu().numpy() for t in final_thetas]
                final_thetas = np.concatenate(final_thetas, axis=0)
            else:
                final_thetas = torch.cat(final_thetas, dim=0)

        return final_thetas

    def get_predictions(self, dataset, to_simplex=True, num_workers=None, to_numpy=True, num_samples: int = 1):
        """
        Predict the labels of the documents in the corpus based on topic proportions.

        Args:
            dataset: a GTMCorpus object
            to_simplex: whether to map the topic distribution to the simplex. If False, the topic distribution is returned in the logit space.
            num_workers: number of workers for the data loaders.
            to_numpy: whether to return the predictions as a numpy array.
            num_samples: number of samples for VAE (only used if ae_type == 'vae').
        """
        if num_workers is None:
            num_workers = self.num_workers

        self.encoder.eval()
        self.predictor.eval()

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
                    if isinstance(value, torch.Tensor):
                        data[key] = value.to(self.device)

                prevalence_covariates = data.get("M_prevalence_covariates", None)
                prediction_covariates = data.get("M_prediction", None)

                # Prepare encoder input
                modality_inputs = {}
                for key in self.encoder_input:
                    mod, view = parse_modality_view(key)
                    x = data["modalities"][mod][view].to(self.device)
                    if prevalence_covariates is not None:
                        x = torch.cat([x, prevalence_covariates], dim=1)
                    modality_inputs[key] = x

                if self.ae_type == "vae":
                    thetas = []
                    for _ in range(num_samples):
                        theta_q, z, _ = self.encoder(modality_inputs)
                        theta_q = theta_q if to_simplex else z
                        thetas.append(theta_q)
                    features = torch.stack(thetas, dim=1).mean(dim=1)
                else:
                    theta_q, z, _ = self.encoder(modality_inputs)
                    features = theta_q if to_simplex else z

                predictions = self.predictor(features, prediction_covariates)
                if self.predictor_type == "classifier":
                    predictions = torch.softmax(predictions, dim=1)

                final_predictions.append(predictions)

            if to_numpy:
                final_predictions = [p.cpu().numpy() for p in final_predictions]
                final_predictions = np.concatenate(final_predictions, axis=0)
            else:
                final_predictions = torch.cat(final_predictions, dim=0)

        return final_predictions

    def get_topic_words(self, l_content_covariates=[], topK=8):
        """
        Get the top words per topic, potentially influenced by content covariates.

        Args:
            l_content_covariates: list of content covariate names to influence the topic-word distribution.
            topK: number of top words to return per topic.
        """
        for key in self.decoder_input:
            if key.endswith("bow"):
                decoder = self.decoders[key]
                id2token = self.id2token.get(key, {})
                break
        else:
            raise ValueError("No BOW decoder found — topic-words not available.")

        decoder.eval()
        with torch.no_grad():
            topic_words = {}
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(self.device)

            for k in l_content_covariates:
                idx = [i for i, l in enumerate(self.content_colnames) if l == k][0]
                idxes[:, (self.n_topics + idx)] += 1

            word_dist = decoder(idxes)
            word_dist = F.softmax(word_dist, dim=1)
            _, indices = torch.topk(word_dist, topK, dim=1)
            indices = indices.cpu().tolist()

            for topic_id in range(self.n_topics):
                topic_words[f"Topic_{topic_id}"] = [
                    id2token.get(idx, f"<UNK_{idx}>") for idx in indices[topic_id]
                ]
        return topic_words

    def get_covariate_words(self, topK=8):
        """
        Get the top words associated to a specific content covariate.

        Args:
            topK: number of top words to return per content covariate.
        """
        for key in self.decoder_input:
            if key.endswith("bow"):  
                decoder = self.decoders[key]
                id2token = self.id2token.get(key, {})
                break
        else:
            raise ValueError("No BOW decoder found — covariate-words not available.")

        decoder.eval()
        with torch.no_grad():
            covariate_words = {}
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(self.device)
            word_dist = decoder(idxes)
            word_dist = F.softmax(word_dist, dim=1)
            _, indices = torch.topk(word_dist, topK, dim=1)
            indices = indices.cpu().tolist()
            for i in range(self.n_topics, self.n_topics + self.content_covariate_size):
                cov_name = self.content_colnames[i - self.n_topics]
                covariate_words[cov_name] = [id2token.get(idx, f"<UNK_{idx}>") for idx in indices[i]]
        return covariate_words

    def get_topic_word_distribution(self, l_content_covariates=[], to_numpy=True):
        """
        Get the topic-word distribution of each topic, potentially influenced by covariates.

        Args:
            l_content_covariates: list with the names of the content covariates to influence the topic-word distribution.
            to_numpy: whether to return the topic-word distribution as a numpy array.
        """
        for key in self.decoder_input:
            if key.endswith("bow"):  
                decoder = self.decoders[key]
                break
        else:
            raise ValueError("No BOW decoder found — topic-word distribution not available.")

        decoder.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(self.device)
            for k in l_content_covariates:
                try:
                    idx = self.content_colnames.index(k)
                    idxes[:, self.n_topics + idx] += 1
                except ValueError:
                    raise ValueError(f"Content covariate '{k}' not found in content_colnames.")
            topic_word_distribution = decoder(idxes)
            topic_word_distribution = F.softmax(topic_word_distribution, dim=1)

        return topic_word_distribution[:self.n_topics, :].cpu().numpy() if to_numpy else topic_word_distribution[:self.n_topics, :]

    def get_covariate_word_distribution(self, to_numpy=True):
        """
        Get the covariate-word distribution of each topic.

        Args:
            to_numpy: whether to return the covariate-word distribution as a numpy array.
        """
        for key in self.decoder_input:
            if key.endswith("bow"):  
                decoder = self.decoders[key]
                break
        else:
            raise ValueError("No BOW decoder found — covariate-word distributions not available.")

        decoder.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(self.device)
            word_dist = decoder(idxes)
            word_dist = F.softmax(word_dist, dim=1)

        return word_dist[self.n_topics:, :].cpu().numpy() if to_numpy else word_dist[self.n_topics:, :]

    def get_top_docs(self, dataset, topic_id=None, return_df=False, topK=1, num_samples: int = 1):
        """
        Get the most representative documents per topic.

        Args:
            dataset: a GTMCorpus object
            topic_id: the topic to retrieve the top documents from. If None, the top documents for all topics are returned.
            return_df: whether to return the top documents as a DataFrame.
            topK: number of top documents to return per topic.
            num_samples: number of samples to draw for VAE inference (ignored for WAE).
        """
        doc_topic_distribution = self.get_doc_topic_distribution(
            dataset, to_simplex=True, num_samples=num_samples
        )

        top_k_indices_df = pd.DataFrame(
            {
                f"Topic_{col}": top_k_indices_column(
                    doc_topic_distribution[:, col], topK
                )
                for col in range(doc_topic_distribution.shape[1])
            }
        )

        if not return_df:
            if topic_id is None:
                for topic_id in range(self.n_topics):
                    for i in top_k_indices_df[f"Topic_{topic_id}"]:
                        print(
                            f"Topic: {topic_id} | Document index: {i} | Topic share: {doc_topic_distribution[i, topic_id]:.4f}"
                        )
                        print(dataset.df["doc"].iloc[i])
                        print("\n")
            else:
                for i in top_k_indices_df[f"Topic_{topic_id}"]:
                    print(
                        f"Topic: {topic_id} | Document index: {i} | Topic share: {doc_topic_distribution[i, topic_id]:.4f}"
                    )
                    print(dataset.df["doc"].iloc[i])
                    print("\n")
        else:
            records = []
            for t_id in range(self.n_topics):
                for i in top_k_indices_df[f"Topic_{t_id}"]:
                    records.append({
                        "topic_id": t_id,
                        "doc_id": i,
                        "topic_share": doc_topic_distribution[i, t_id],
                        "doc": dataset.df["doc"].iloc[i]
                    })
            df = pd.DataFrame.from_records(records)
            if topic_id is not None:
                df = df[df["topic_id"] == topic_id].reset_index(drop=True)
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

        for key in self.decoder_input:
            if key.endswith("bow"):
                id2token = self.id2token.get(key, {})
                break

        topic_word_distribution = self.get_topic_word_distribution(
            content_covariates, to_numpy=False
        )
        vals, indices = torch.topk(topic_word_distribution, topK, dim=1)
        vals = vals.cpu().tolist()
        indices = indices.cpu().tolist()
        topic_words = [id2token[idx] for idx in indices[topic_id]]
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
        num_samples: int = 1,
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

        matrix = self.get_doc_topic_distribution(dataset, to_simplex=True, num_samples=num_samples)
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
        num_samples: int = 1
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
        doc_topic_dist = self.get_doc_topic_distribution(dataset, to_simplex=True, num_samples=num_samples)
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

    def estimate_effect(
        self,
        dataset,
        to_simplex: bool = True,
        n_samples: int = 20,
        topic_ids: Optional[List[int]] = None,
        progress_bar: bool = True,
    ) -> pd.DataFrame:
        """
        GLM estimates and associated standard errors of the doc-topic prior conditional on the prevalence covariates.

        Uncertainty is computed using the method of composition:
        - Sample topic proportions from the prior or variational posterior
        - Fit OLS: topic_proportion ~ covariates
        - Repeat n_samples times
        - Report mean and standard deviation of coefficients

        Returns:
            pd.DataFrame with columns: ['topic', 'covariate', 'mean', 'sd']
        """
        X = dataset.M_prevalence_covariates  # (N, C)
        covariate_names = dataset.prevalence_colnames

        topic_ids = topic_ids if topic_ids is not None else list(range(self.n_topics))
        iterator = tqdm(range(n_samples)) if progress_bar else range(n_samples)

        # Store regression coefficients per topic
        coefs_by_topic = {k: [] for k in topic_ids}
        model = LinearRegression(fit_intercept=False)

        for _ in iterator:
            # Sample topic proportions
            if self.doc_topic_prior == "dirichlet":
                Y = self.prior.sample(N=X.shape[0], M_prevalence_covariates=X).cpu().numpy()
            elif self.ae_type == "vae":
                Y = self.get_doc_topic_distribution(
                    dataset,
                    to_simplex=to_simplex,
                    to_numpy=True,
                    num_samples=1,
                )
            else:
                Y = self.prior.sample(N=X.shape[0], M_prevalence_covariates=X, to_simplex=to_simplex).cpu().numpy()

            for k in topic_ids:
                model.fit(X, Y[:, k])
                coefs_by_topic[k].append(model.coef_.copy())

        # Aggregate results
        records = []
        for k in topic_ids:
            all_coefs = np.stack(coefs_by_topic[k], axis=0)  # (n_samples, n_covariates)
            mean = all_coefs.mean(axis=0)
            std = all_coefs.std(axis=0)
            for i, cov in enumerate(covariate_names):
                records.append({
                    "topic": k,
                    "covariate": cov,
                    "mean": mean[i],
                    "sd": std[i],
                })

        return pd.DataFrame(records)

    def save_model(self, save_name):
        encoder_state_dict = self.encoder.state_dict()
        decoders_state_dict = {k: d.state_dict() for k, d in self.decoders.items()}
        predictor_state_dict = self.predictor.state_dict() if self.labels_size != 0 else None
        optimizer_state_dict = self.optimizer.state_dict()

        all_vars = vars(self)

        checkpoint = {}
        for key, value in all_vars.items():
            if key not in ["encoder", "decoders", "predictor", "optimizer"]:
                checkpoint[key] = value

        checkpoint["encoder"] = encoder_state_dict
        checkpoint["decoders"] = decoders_state_dict
        if self.labels_size != 0:
            checkpoint["predictor"] = predictor_state_dict
        checkpoint["optimizer"] = optimizer_state_dict

        torch.save(checkpoint, save_name)

    def load_model(self, ckpt):
        """
        Helper function to load a GTM model.
        """
        ckpt = torch.load(ckpt, map_location=self.device, weights_only=False)

        for key, value in ckpt.items():
            if key not in ["encoder", "decoders", "predictor", "optimizer"]:
                setattr(self, key, value)

        self.encoder.load_state_dict(ckpt["encoder"])

        for key, state_dict in ckpt["decoders"].items():
            self.decoders[key].load_state_dict(state_dict)

        if self.labels_size != 0 and "predictor" in ckpt:
            if not hasattr(self, "predictor"):
                predictor_dims = [self.n_topics + self.prediction_covariate_size] + \
                                self.predictor_hidden_layers + [self.labels_size]
                self.predictor = Predictor(
                    predictor_dims=predictor_dims,
                    predictor_non_linear_activation=self.predictor_non_linear_activation,
                    predictor_bias=self.predictor_bias,
                    dropout=self.dropout,
                ).to(self.device)
            self.predictor.load_state_dict(ckpt["predictor"])

        if not hasattr(self, "optimizer"):
            all_params = list(self.encoder.parameters()) + list(self.decoders.parameters())
            if self.labels_size != 0:
                all_params += list(self.predictor.parameters())
            self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)

        self.optimizer.load_state_dict(ckpt["optimizer"])

    def to(self, device):
        """
        Move the model to a different device.
        """
        self.encoder.to(device)
        self.decoders.to(device)
        self.prior.to(device)
        if self.labels_size != 0:
            self.predictor.to(device)
        self.device = device