#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn.functional as F
from autoencoders import AutoEncoderMLP, AutoEncoderSAGE
from predictors import Predictor
from priors import DirichletPrior, LogisticNormalPrior
from torch.utils.data import DataLoader
from utils import compute_mmd_loss, top_k_indices_column
from wordcloud import WordCloud

# TO-DO:
# integrate with OCTIS


class GTM:
    """
    Wrapper class for the Generalized Topic Model.
    """

    def __init__(
        self,
        train_data,
        test_data=None,
        n_topics=20,
        doc_topic_prior="dirichlet",
        update_prior=False,
        alpha=0.1,
        prevalence_covariates_regularization=0,
        tol=0.001,
        encoder_input="bow",
        encoder_hidden_layers=[1024, 512],
        encoder_non_linear_activation="relu",
        encoder_bias=True,
        decoder_type="mlp",
        decoder_hidden_layers=[300],
        decoder_non_linear_activation=None,
        decoder_bias=False,
        decoder_estimate_interactions=False,
        decoder_sparsity_regularization=0.1,
        predictor_type=None,
        predictor_hidden_layers=[],
        predictor_non_linear_activation=None,
        predictor_bias=False,
        num_epochs=10,
        num_workers=4,
        batch_size=256,
        learning_rate=1e-3,
        dropout=0.2,
        print_every=10,
        log_every=5,
        w_prior=None,
        w_pred_loss=1,
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
            prevalence_covariates_regularization: regularization parameter for the logistic normal prior (only used if update_prior=True)
            tol: tolerance threshold to stop the MLE of the Dirichlet prior (only used if update_prior=True)
            encoder_input: input to the encoder. Either 'bow' or 'embeddings'. 'bow' is a simple Bag-of-Words representation of the documents. 'embeddings' is the representation from a pre-trained embedding model (e.g. GPT, BERT, GloVe, etc.).
            encoder_hidden_layers: list with  the size of the hidden layers for the encoder.
            encoder_non_linear_activation: non-linear activation function for the encoder.
            encoder_bias: whether to use bias in the encoder.
            decoder_type: type of decoder. Either 'mlp' or 'sage'. 'mlp' is an arbitrarily complex Multilayer Perceptron. 'sage' is a Sparse Additive Generative Model.
            decoder_hidden_layers: list with the size of the hidden layers for the decoder (only used with decoder_type='mlp').
            decoder_non_linear_activation: non-linear activation function for the decoder (only used with decoder_type='mlp').
            decoder_bias: whether to use bias in the decoder (only used with decoder_type='mlp').
            decoder_sparsity_regularization: regularization parameter for the decoder (only used with decoder_type='sage').
            predictor_type: type of predictor. Either 'classifier' or 'regressor'. 'classifier' predicts a categorical variable, 'regressor' predicts a continuous variable.
            predictor_hidden_layers: list with the size of the hidden layers for the predictor.
            predictor_non_linear_activation: non-linear activation function for the predictor.
            predictor_bias: whether to use bias in the predictor.
            num_epochs: number of epochs to train the model.
            num_workers: number of workers for the data loaders.
            batch_size: batch size for training.
            learning_rate: learning rate for training.
            dropout: dropout rate for training.
            print_every: number of batches between each print.
            log_every: number of epochs between each checkpoint.
            patience: number of epochs to wait before stopping the training if the validation or training loss does not improve.
            delta: threshold to stop the training if the validation or training loss does not improve.
            w_prior: parameter to control the tightness of the encoder output with the document-topic prior. If set to None, w_prior is chosen automatically.
            w_pred_loss: parameter to control the weight given to the prediction task in the likelihood. Default is 1.
            ckpt: checkpoint to load the model from.
            device: device to use for training.
            seed: random seed.

        References:
            - Eisenstein, J., Ahmed, A., & Xing, E. P. (2011). Sparse additive generative models of text. In Proceedings of the 28th international conference on machine learning (ICML-11) (pp. 1041-1048).
            - Nan, F., Ding, R., Nallapati, R., & Xiang, B. (2019). Topic modeling with wasserstein autoencoders. arXiv preprint arXiv:1907.12374.
        """

        self.n_topics = n_topics
        self.doc_topic_prior = doc_topic_prior
        self.update_prior = update_prior
        self.alpha = alpha
        self.prevalence_covariates_regularization = prevalence_covariates_regularization
        self.tol = tol
        self.encoder_input = encoder_input
        self.encoder_hidden_layers = encoder_hidden_layers
        self.encoder_non_linear_activation = encoder_non_linear_activation
        self.encoder_bias = encoder_bias
        self.decoder_type = decoder_type
        self.decoder_hidden_layers = decoder_hidden_layers
        self.decoder_non_linear_activation = decoder_non_linear_activation
        self.decoder_bias = decoder_bias
        self.decoder_estimate_interactions = decoder_estimate_interactions
        self.decoder_sparsity_regularization = decoder_sparsity_regularization
        self.predictor_type = predictor_type
        self.predictor_hidden_layers = predictor_hidden_layers
        self.predictor_non_linear_activation = predictor_non_linear_activation
        self.predictor_bias = predictor_bias
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.dropout = dropout
        self.print_every = print_every
        self.log_every = log_every
        self.patience = patience
        self.delta = delta
        self.w_prior = w_prior
        self.w_pred_loss = w_pred_loss
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            np.random.seed(seed)

        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            # support macos GPU
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")

        bow_size = train_data.M_bow.shape[1]
        self.bow_size = bow_size

        if train_data.prevalence is not None:
            prevalence_covariate_size = train_data.M_prevalence_covariates.shape[1]
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

        encoder_dims = [self.input_size + prevalence_covariate_size + labels_size]
        encoder_dims.extend(encoder_hidden_layers)
        encoder_dims.extend([n_topics])

        decoder_dims = [n_topics + content_covariate_size]
        decoder_dims.extend(decoder_hidden_layers)
        decoder_dims.extend([bow_size])

        if decoder_type == "mlp":
            self.AutoEncoder = AutoEncoderMLP(
                encoder_dims=encoder_dims,
                encoder_non_linear_activation=encoder_non_linear_activation,
                encoder_bias=encoder_bias,
                decoder_dims=decoder_dims,
                decoder_non_linear_activation=decoder_non_linear_activation,
                decoder_bias=decoder_bias,
                dropout=dropout,
            ).to(self.device)
        elif decoder_type == "sage":
            self.AutoEncoder = AutoEncoderSAGE(
                encoder_dims=encoder_dims,
                encoder_non_linear_activation=encoder_non_linear_activation,
                encoder_bias=encoder_bias,
                dropout=dropout,
                bow_size=bow_size,
                content_covariate_size=content_covariate_size,
                estimate_interactions=decoder_estimate_interactions,
                log_word_frequencies=train_data.log_word_frequencies,
                l1_beta_reg=decoder_sparsity_regularization,
                l1_beta_c_reg=decoder_sparsity_regularization,
                l1_beta_ci_reg=decoder_sparsity_regularization,
            ).to(self.device)

        if doc_topic_prior == "dirichlet":
            self.prior = DirichletPrior(
                prevalence_covariate_size,
                n_topics,
                alpha,
                prevalence_covariates_regularization,
                tol,
                device=self.device,
            )
        elif doc_topic_prior == "logistic_normal":
            self.prior = LogisticNormalPrior(
                prevalence_covariate_size,
                n_topics,
                prevalence_covariates_regularization,
                device=self.device,
            )

        if labels_size != 0:
            predictor_dims = [n_topics+prediction_covariate_size]
            predictor_dims.extend(predictor_hidden_layers)
            predictor_dims.extend([n_labels])
            self.predictor = Predictor(
                predictor_dims=predictor_dims,
                predictor_non_linear_activation=predictor_non_linear_activation,
                predictor_bias=predictor_bias,
                dropout=dropout,
            ).to(self.device)
        # for tracking the mean train loss
        self.mean_val_loss = []
        self.mean_train_loss = []

        self.train(
            train_data,
            test_data,
            batch_size,
            learning_rate,
            num_epochs,
            num_workers,
            log_every,
            print_every,
            w_prior,
            w_pred_loss,
            ckpt,
        )

    def train(
        self,
        train_data,
        test_data,
        batch_size,
        learning_rate,
        num_epochs,
        num_workers,
        log_every,
        print_every,
        w_prior,
        w_pred_loss,
        ckpt,
    ):
        """
        Train the model.
        """

        train_data_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        if test_data is not None:
            test_data_loader = DataLoader(
                test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )

        if self.labels_size != 0:
            optimizer = torch.optim.Adam(
                list(self.AutoEncoder.parameters()) + list(self.predictor.parameters()),
                lr=learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                self.AutoEncoder.parameters(), lr=learning_rate
            )

        if self.decoder_type == "sage":
            n_train = train_data.M_bow.shape[0]
            l1_beta = (
                0.5
                * np.ones([self.bow_size, self.n_topics], dtype=np.float32)
                / float(n_train)
            )
            if self.content_covariate_size != 0:
                l1_beta_c = (
                    0.5
                    * np.ones(
                        [self.bow_size, self.content_covariate_size], dtype=np.float32
                    )
                    / float(n_train)
                )
                l1_beta_ci = (
                    0.5
                    * np.ones(
                        [self.bow_size, self.n_topics * self.content_covariate_size],
                        dtype=np.float32,
                    )
                    / float(n_train)
                )
            else:
                l1_beta_c = None
                l1_beta_ci = None
        else:
            l1_beta = None
            l1_beta_c = None
            l1_beta_ci = None

        if ckpt:
            ckpt = torch.load(ckpt)
            self.load_model(ckpt)
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
        else:
            start_epoch = 0

        counter = 0
        best_loss = np.Inf
        best_epoch = -1
        self.save_model('../ckpt/best_model.ckpt', optimizer, epoch=-1)

        for epoch in range(start_epoch, num_epochs):
            self.epoch(
                train_data_loader,
                optimizer,
                epoch,
                print_every,
                w_prior,
                w_pred_loss,
                l1_beta,
                l1_beta_c,
                l1_beta_ci,
                validation=False,
            )

            if test_data is not None:
                self.epoch(
                    test_data_loader,
                    optimizer,
                    epoch,
                    print_every,
                    w_prior,
                    w_pred_loss,
                    l1_beta,
                    l1_beta_c,
                    l1_beta_ci,
                    validation=True,
                )

            if (epoch + 1) % log_every == 0:
                save_name = f'../ckpt/GTM_K{self.n_topics}_{self.doc_topic_prior}_{self.decoder_type}_{self.predictor_type}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_{epoch+1}.ckpt'

                autoencoder_state_dict = self.AutoEncoder.state_dict()
                if self.labels_size != 0:
                    predictor_state_dict = self.predictor.state_dict()
                else:
                    predictor_state_dict = None

                checkpoint = {
                    "Prior": self.prior,
                    "AutoEncoder": autoencoder_state_dict,
                    "Predictor": predictor_state_dict,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "param": {
                        "input_dim": self.input_size,
                        "n_topics": self.n_topics,
                        "doc_topic_prior": self.doc_topic_prior,
                        "dropout": self.dropout,
                    },
                }
                torch.save(checkpoint, save_name)
                print(
                    "\n".join(
                        [
                            "{}: {}".format(k, str(v["words"]))
                            for k, v in self.get_topic_words().items()
                        ]
                    )
                )
                print("\n")

            if self.update_prior:
                posterior_theta = self.get_doc_topic_distribution(train_data)
                self.prior.update_parameters(
                    posterior_theta, train_data.M_prevalence_covariates
                )

            print(self.prior.lambda_)

            if self.decoder_type == "sage":
                (
                    l1_beta,
                    l1_beta_c,
                    l1_beta_ci,
                ) = self.AutoEncoder.update_jeffreys_priors(n_train)

    def epoch(
        self,
        data_loader,
        optimizer,
        epoch,
        print_every,
        w_prior,
        w_pred_loss,
        l1_beta,
        l1_beta_c,
        l1_beta_ci,
        validation=False,
    ):
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
                optimizer.zero_grad()

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
            x_recon, theta_q = self.AutoEncoder(
                x_input, prevalence_covariates, content_covariates, target_labels
            )
            reconstruction_loss = F.cross_entropy(x_recon, x_bows)

            # Get prior on theta and compute regularization loss
            theta_prior = self.prior.sample(
                N=x_input.shape[0],
                M_prevalence_covariates=prevalence_covariates,
                epoch=epoch,
            ).to(self.device)
            mmd_loss = compute_mmd_loss(theta_q, theta_prior, device=self.device, t=0.1)

            if self.w_prior is None:
                mean_length = torch.sum(x_bows) / x_bows.shape[0]
                vocab_size = x_bows.shape[1]
                w_prior = mean_length * np.log(vocab_size)
            else:
                w_prior = self.w_prior

            # Add regularization to induce sparsity in the topic-word-covariate distributions
            if self.decoder_type == "sage":
                decoder_sparsity_loss = self.AutoEncoder.sparsity_loss(
                    l1_beta, l1_beta_c, l1_beta_ci, self.device
                )
            else:
                decoder_sparsity_loss = 0

            # Predict labels and compute prediction loss
            if target_labels is not None:
                predictions = self.predictor(theta_q)
                # y_pred_probs = torch.softmax(predictions, dim=1)
                # print(y_pred_probs)
                if self.predictor_type == "classifier":
                    target_labels = target_labels.squeeze().to(torch.int64)
                if self.predictor_type == "classifier":
                    prediction_loss = F.cross_entropy(predictions, target_labels)
                elif self.predictor_type == "regressor":
                    prediction_loss = F.mse_loss(predictions, target_labels)
            else:
                prediction_loss = 0

            # Total loss
            loss = (
                reconstruction_loss
                + mmd_loss * w_prior
                + prediction_loss * w_pred_loss
                + decoder_sparsity_loss
            )

            if not validation:
                loss.backward()
                optimizer.step()

            epochloss_lst.append(loss.item() / len(x_input))

            if (iter + 1) % print_every == 0:
                if validation:
                    print(
                        f"Epoch {(epoch+1):>3d}\tIter {(iter+1):>4d}\tValidation Loss:{loss.item()/len(x_input):<.7f}\nRec Loss:{reconstruction_loss.item()/len(x_input):<.7f}\nMMD Loss:{mmd_loss.item()*w_prior/len(x_input):<.7f}\nSparsity Loss:{decoder_sparsity_loss/len(x_input):<.7f}\nPred Loss:{prediction_loss*w_pred_loss/len(x_input):<.7f}\n"
                    )
                else:
                    print(
                        f"Epoch {(epoch+1):>3d}\tIter {(iter+1):>4d}\tTraining Loss:{loss.item()/len(x_input):<.7f}\nRec Loss:{reconstruction_loss.item()/len(x_input):<.7f}\nMMD Loss:{mmd_loss.item()*w_prior/len(x_input):<.7f}\nSparsity Loss:{decoder_sparsity_loss/len(x_input):<.7f}\nPred Loss:{prediction_loss*w_pred_loss/len(x_input):<.7f}\n"
                    )

        if validation:
            print(
                f"\nEpoch {(epoch+1):>3d}\tMean Validation Loss:{sum(epochloss_lst)/len(epochloss_lst):<.7f}\n"
            )
            self.mean_val_loss.append(sum(epochloss_lst) / len(epochloss_lst))
        else:
            print(
                f"\nEpoch {(epoch+1):>3d}\tMean Training Loss:{sum(epochloss_lst)/len(epochloss_lst):<.7f}\n"
            )
            self.mean_train_loss.append(sum(epochloss_lst) / len(epochloss_lst))

        return sum(epochloss_lst)

    def get_doc_topic_distribution(self, dataset, to_simplex=True, num_workers=4, to_numpy=True):
        """
        Get the topic distribution of each document in the corpus.

        Args:
            dataset: a GTMCorpus object
            to_simplex: whether to map the topic distribution to the simplex. If False, the topic distribution is returned in the logit space.
        """
        with torch.no_grad():
            data_loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
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
                target_labels = data.get("M_labels", None)
                if self.encoder_input == "bow":
                    x_input = bows
                elif self.encoder_input == "embeddings":
                    x_input = embeddings
                _, thetas = self.AutoEncoder(
                    x_input, prevalence_covariates, content_covariates, target_labels
                )
                thetas = thetas.cpu().numpy()
                final_thetas.append(thetas)

        return np.concatenate(final_thetas, axis=0)

    def get_topic_words(self, content_covariates=None, topK=5):
        """
        Get the top words per topic, potentially influenced by content covariates.
        """
        self.AutoEncoder.eval()
        with torch.no_grad():
            if self.decoder_type == "mlp":
                topic_words = {}
                idxes = torch.eye(self.n_topics + self.content_covariate_size).to(
                    self.device
                )
            if self.decoder_type == 'mlp':
                topic_words = {}
                idxes = torch.eye(self.n_topics+self.content_covariate_size).to(self.device)
                word_dist = self.AutoEncoder.decode(idxes)
                word_dist = F.softmax(word_dist, dim=1)
                vals, indices = torch.topk(word_dist, topK, dim=1)
                vals = vals.cpu().tolist()
                indices = indices.cpu().tolist()
                for topic_id in range(self.n_topics + self.content_covariate_size):
                    if topic_id < self.n_topics:
                        topic_words["Topic_{}".format(topic_id)] = {
                            "words": [self.id2token[idx] for idx in indices[topic_id]],
                            "values": vals[topic_id],
                        }
                    else:
                        i = topic_id - self.n_topics
                        topic_words[self.content_colnames[i]] = {
                            "words": [self.id2token[idx] for idx in indices[topic_id]],
                            "values": vals[topic_id],
                        }
            elif self.decoder_type == "sage":
                topic_words = {}
                word_dist = F.softmax(self.AutoEncoder.beta.weight.T, dim=1)
                vals, indices = torch.topk(word_dist, topK, dim=1)
                vals = vals.cpu().tolist()
                indices = indices.cpu().tolist()
                for topic_id in range(self.n_topics):
                    topic_words["Topic_{}".format(topic_id)] = {
                        "words": [self.id2token[idx] for idx in indices[topic_id]],
                        "values": vals[topic_id],
                    }
                vals, indices = (
                    torch.topk(self.AutoEncoder.beta.bias, k=topK, dim=0)
                    .indices.cpu()
                    .numpy()
                    .flatten()
                )
                topic_words["Common_words:"] = {
                    "words": [self.id2token[idx] for idx in indices[topic_id]],
                    "values": vals[topic_id],
                }

        return topic_words

    def get_topic_word_distribution(self):
        """
        Get the topic-word distribution of each baseline topic.
        """
        self.AutoEncoder.eval()
        with torch.no_grad():
            if self.decoder_type == "mlp":
                idxes = torch.eye(self.n_topics + self.content_covariate_size).to(
                    self.device
                )
                topic_word_distribution = self.AutoEncoder.decode(idxes)
                topic_word_distribution = F.softmax(topic_word_distribution, dim=1)
            elif self.decoder_type == "sage":
                topic_word_distribution = []
                topic_word_distribution = F.softmax(
                    self.AutoEncoder.beta.weight.T, dim=1
                )
        return topic_word_distribution.cpu().detach().numpy()[0 : self.n_topics, :]

    def get_top_docs(self, dataset, topic_id=None, return_df=False, topK=1):
        """
        Get the most representative documents per topic.
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

    def plot_wordcloud(
        self,
        topic_id,
        content_covariates=None,
        topK=100,
        output_path=None,
        wordcloud_args={},
    ):
        """
        Returns a wordcloud representation per topic.
        """
        topic_word_distribution = self.get_topic_words(content_covariates, topK)
        temp = topic_word_distribution["Topic_{}".format(topic_id)]
        d = {}
        for i, w in enumerate(temp["words"]):
            d[w] = temp["values"][i]
        wordcloud = WordCloud(**wordcloud_args).generate_from_frequencies(d)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        if output_path is not None:
            plt.savefig(output_path)

    def estimate_effect(self, dataset, n_samples=20, topic_ids=None):
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

        dict_of_params = {"Topic_{}".format(k): [] for k in range(self.n_topics)}
        for i in range(n_samples):
            Y = self.prior.sample(X.shape[0], X).cpu().numpy()
            for k in iterator:
                model = sm.OLS(Y[:, k], X)
                results = model.fit()
                dict_of_params["Topic_{}".format(k)].append(np.array([results.params]))

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

    def get_topic_correlations(self):
        """
        Plot correlations between topics for a logistic normal prior.
        """
        # Represent as a standard variance-covariance matrix
        # See https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
        sigma = pd.DataFrame(self.prior.sigma.detach().cpu().numpy())
        mask = np.zeros_like(sigma, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        sigma[mask] = np.nan
        p = (
            sigma.style.background_gradient(cmap="coolwarm", axis=None, vmin=-1, vmax=1)
            .highlight_null(color="#f1f1f1")  # Color NaNs grey
            .format(precision=2)
        )
        return p

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

    def save_model(self, save_name, optimizer, epoch):
        autoencoder_state_dict = self.AutoEncoder.state_dict()
        if self.labels_size != 0:
            predictor_state_dict = self.predictor.state_dict()
        else:
            predictor_state_dict = None

        optimizer_state_dict = optimizer.state_dict()

        checkpoint = {
            "Prior": self.prior,
            "AutoEncoder": autoencoder_state_dict,
            "Predictor": predictor_state_dict,
            "optimizer": optimizer_state_dict,
            "epoch": epoch,
            "param": {
                "input_dim": self.input_size,
                "n_topics": self.n_topics,
                "doc_topic_prior": self.doc_topic_prior,
                "dropout": self.dropout
            }
        }
        torch.save(checkpoint,save_name)

    def load_model(self, ckpt):
        """
        Helper function to load a GTM model.
        """
        self.AutoEncoder.load_state_dict(ckpt["AutoEncoder"])
        self.prior = ckpt["Prior"]
        if self.labels_size != 0:
            self.predictor.load_state_dict(ckpt["Predictor"])
