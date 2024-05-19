import pandas as pd
import sys
from gtm.corpus import GTMCorpus
from gtm.gtm import GTM
import pickle as p
import os
from compute_metrics import compute
import sys

MODEL_NAME = 'intfloat/multilingual-e5-large'

# redirect stdout to file, and also print to console
sys.stdout = open('gtm_wiki_task1_redo_{}.log'.format(MODEL_NAME.replace('/', '-')), 'w')

def load_examples(language='en'):
  df = pd.read_csv('./dataset/docs_original_{}.txt'.format(language), header=None, delimiter='\t')
  df.columns = ['index', 'doc_clean']
  df['language'] = language
  # df = df.head(1000)
  return df

df_en = load_examples('en')
df_zh = load_examples('zh')

# # first 100 English examples
# df_en = df_en.head(100)
# # first 120 Chinese examples
# df_zh = df_zh.head(120)

print('English examples = {}'.format(len(df_en)))
print('Chinese examples = {}'.format(len(df_zh)))

# combine the two df
num_en = len(df_en)
num_zh = len(df_zh)
# English always comes the first
df = pd.concat([df_en, df_zh]).reset_index()
print('Total examples = {}'.format(len(df)))

def create_dataset(language='en'):
  if not os.path.exists('train_dataset_{}-{}.pkl'.format(MODEL_NAME.split('/')[1], language)):
    print('Loading examples for {}'.format(language))
    train_dataset = GTMCorpus(
      df,
      embeddings_type='SentenceTransformer',
      vectorizer_args = {'ngram_range':(1, 1), 'max_df':0.99, 'min_df':0.001, 'stop_words':'english'},
      sbert_model_to_load=MODEL_NAME,
      content=None,
      prevalence=None,
      batch_size=128,
      max_seq_length=512,
      num_en=num_en,
      num_zh=num_zh,
      language=language)

    print('Saving train_dataset_{}-{}.pkl'.format(MODEL_NAME.split('/')[1], language))
    with open('train_dataset_{}-{}.pkl'.format(MODEL_NAME.split('/')[1], language), 'wb') as f:
      p.dump(train_dataset, f)
  else:
    with open('train_dataset_{}-{}.pkl'.format(MODEL_NAME.split('/')[1], language), 'rb') as f:
      train_dataset = p.load(f)
  return train_dataset

train_dataset_en = create_dataset('en')
train_dataset_zh = create_dataset('zh')

# exit()

tm_en = GTM(
    train_dataset_en,
    n_topics=6,
    doc_topic_prior='dirichlet',
    update_prior=False,
    encoder_hidden_layers=[],
    decoder_hidden_layers=[256],
    learning_rate=1e-3,
    num_workers=0,
    patience=3,
    num_epochs=1000,
    encoder_input='bow',
    ckpt_folder='./ckpt_task1_redo_en',
    # ckpt='./ckpt_task1_redo_en/best_model.ckpt',
)
print('Computing metrics for English... BOW')
compute(tm_en, train_dataset_en, './output_task1_BOW_en_{}'.format(MODEL_NAME))

tm_zh = GTM(
    train_dataset_zh,
    n_topics=6,
    doc_topic_prior='dirichlet',
    update_prior=False,
    encoder_hidden_layers=[],
    decoder_hidden_layers=[256],
    learning_rate=1e-3,
    num_workers=0,
    patience=3,
    num_epochs=1000,
    encoder_input='bow',
    ckpt_folder='./ckpt_task1_redo_zh',
    # ckpt='./ckpt_task1_redo_zh/best_model.ckpt',
)
print('Computing metrics for Chinese... BOW')
compute(tm_zh, train_dataset_zh, './output_task1_BOW_zh_{}'.format(MODEL_NAME))

print('Computing metrics for Chinese... BOW, zh2en')
compute(tm_zh, train_dataset_en, './output_task1_BOW_zh_zh2en_{}'.format(MODEL_NAME))

print('Computing metrics for English... BOW, en2zh')
compute(tm_en, train_dataset_zh, './output_task1_BOW_zh_en2zh_{}'.format(MODEL_NAME))


tm_en = GTM(
    train_dataset_en,
    n_topics=6,
    doc_topic_prior='dirichlet',
    update_prior=False,
    encoder_hidden_layers=[],
    decoder_hidden_layers=[256],
    learning_rate=1e-3,
    num_workers=0,
    patience=3,
    num_epochs=1000,
    encoder_input='embeddings',
    ckpt_folder='./ckpt_task1_redo_en_emb',
    # ckpt='./ckpt_task1_redo_en_emb/best_model.ckpt',
)

print('Computing metrics for English... Embeddings')
compute(tm_en, train_dataset_en, './output_task1_redo_en_emb')

tm_zh = GTM(
    train_dataset_zh,
    n_topics=6,
    doc_topic_prior='dirichlet',
    update_prior=False,
    encoder_hidden_layers=[],
    decoder_hidden_layers=[256],
    learning_rate=1e-3,
    num_workers=0,
    patience=3,
    num_epochs=1000,
    encoder_input='embeddings',
    ckpt_folder='./ckpt_task1_redo_zh_emb',
    # ckpt='./ckpt_task1_redo_zh_emb/best_model.ckpt',
)

print('Computing metrics for Chinese... Embeddings')
compute(tm_zh, train_dataset_zh, './output_task1_redo_zh_emb_{}'.format(MODEL_NAME))

print('Computing metrics for Chinese... Embeddings, zh2en')
compute(tm_zh, train_dataset_en, './output_task1_redo_zh_emb_zh2en_{}'.format(MODEL_NAME))

print('Computing metrics for English... Embeddings, en2zh')
compute(tm_en, train_dataset_zh, './output_task1_redo_zh_emb_en2zh_{}'.format(MODEL_NAME))
