# Generalized Topic Model (GTM)

**GTM** is a **neural topic modeling framework** for large multimodal/multilingual corpora.

It supports:

- **Exploratory analysis** of latent topics  
- **Supervised prediction** using topic features  
- **Causal modeling** via structured priors and metadata

## Key Features

- **Multilingual and multimodal support**
- **Flexible metadata handling**:
  - `prevalence`: influences topic choice  
  - `content`: alters topic content conditioned on topic  
  - `labels`: for classification or regression tasks  
  - `prediction`: additional predictors for labels
- **Input representations**:
  - Document embeddings
  - Word frequency (BoW)

---

## 🚀 Getting Started

### 1. Build the Dataset with `GTMCorpus()`

Prepares your corpus and metadata.

#### ✅ Supported Inputs:

- **Metadata** (optional):  
  - `prevalence`, `content`, `labels`, `prediction`
- **Multimodal views**:
  
```python
modalities = {
    "text": {
        "column": "doc_clean",
        "views": {
            "bow": {
                "type": "bow",
                "vectorizer": CountVectorizer()
            }
        }
    },
    "image": {
        "column": "image_path",
        "views": {
            "embedding": {
                "type": "embedding",
                "embed_fn": my_image_embedder
            }
        }
    }
}
```

---

### 2. Train the GTM Model

```python
model = GTM(...)
```

#### 🔧 Core Options:

- `n_topics`: number of latent topics  
- `doc_topic_prior`:  
  - `"dirichlet"` (sparse, interpretable)  
  - `"logistic_normal"` (flexible, use with `vae`)

#### ⚖️ Loss Weights:

- `w_prior`: how much metadata influences topics  
- `w_pred_loss`: weight of supervised loss (if using `labels`)

#### 📐 Structured Priors:

- Set `update_prior=True` to condition topic priors on `prevalence` covariates

#### 🧬 Autoencoder Type:

- `"wae"`: Wasserstein Autoencoder (default, stable)  
- `"vae"`: Variational Autoencoder  
  - Use with `doc_topic_prior="logistic_normal"`

#### 🔁 KL Annealing (VAE only):

Prevents posterior collapse and encourages meaningful topics:

```python
kl_annealing_start = 0
kl_annealing_end = 1000
kl_annealing_max_beta = 1.0
```

---

### 3. Explore and Analyze Topics

#### 📝 Topic Inspection:

- `get_topic_words()` — top words per topic  
- `get_covariate_words()` — word shifts by `content` covariates  
- `get_top_docs()` — representative docs per topic

#### 📈 Metadata Effects:

- `estimate_effect()` — topic prevalence regression (linear)

#### 🖼️ Visualizations:

- `plot_topic_word_distribution()` — word clouds / bar plots  
- `visualize_docs()` — 2D projection (UMAP, t-SNE, PCA)  
- `visualize_words()` — semantic word embeddings  
- `visualize_topics()` — semantic topic embeddings

#### 🎯 Supervised Prediction:

- `get_predictions()` — returns classification or regression outputs (if `labels` were used)

---

## 📚 Tutorials

Get started with example notebooks in [`notebooks/`](notebooks/).

The dataset used in these notebooks can be downloaded [here](https://www.dropbox.com/scl/fi/ojshavj5azk4jt7a4p3ap/us_congress_speeches_sample.csv?rlkey=x3x86kc9pb94kuu1c8yze5u3l&st=awtc4wr2&dl=1) and should be placed in the `data` folder.

---

## 📖 References

- [**Deep Latent Variable Models for Unstructured Data** (PDF)](https://www.dropbox.com/scl/fi/c30hibel8ad93owfiz2lh/Deep_Latent_Variable_Models_for_Unstructured_Data.pdf?rlkey=xn9u9og0d0a603i4b7j4i511a&st=pisq7110&dl=0)  
  *Germain Gauthier, Philine Widmer, and Elliott Ash*

- [**generalized_topic_models: A Python Package to Estimate Neural Topic Models** (PDF)](https://www.dropbox.com/scl/fi/g8j1wec3uy7g1w37gapdc/GTM_JSS_draft.pdf?rlkey=pdfmylxxcs5r6w2f0hilb74xo&st=vhvci1kz&dl=0)  
  *Germain Gauthier, Philine Widmer, and Elliott Ash*

---

## ⚠️ Disclaimer

This package is under active development 🚧 — feedback and contributions are welcome!
