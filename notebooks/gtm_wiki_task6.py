import pandas as pd
import sys

sys.path.append('../gtm/')
from corpus_task6 import GTMCorpus
from gtm_customized import GTM
import pickle as p
import argparse
import numpy as np
import os

language = 'en-zh'

def load_examples(language='en'):
  # read the documents
  df_docs = pd.read_csv('../data/wiki_shorts/{}/corpus/docs.txt'.format(language), header=None, delimiter='\t')
  df_docs.columns = ['doc_clean']
  # read the labels as integers
  df_categories = pd.read_csv('../data/wiki_shorts/{}/labels.txt'.format(language), header=None, delimiter='\t')
  df_categories.columns = ['category']
  # combine the documents and labels
  df = pd.concat([df_docs, df_categories], axis=1)
  return df, df_docs, df_categories

# try mixedbread-ai/mxbai-embed-large-v1 or sentence-transformers/all-mpnet-base-v2
def create_dataset(model_name='sentence-transformers/all-mpnet-base-v2', batch_size=64, max_seq_length=512):
  if not os.path.exists('train_dataset_{}-{}.pkl'.format(model_name.replace('/', '_'), 'en-zh')):
    df_en, df_docs_en, df_categories_en = load_examples('en')
    df_zh, df_docs_zh, df_categories_zh = load_examples('zh')
    df_en['language'] = 'en'
    df_zh['language'] = 'zh'

    # merge df_en and df_zh
    df = pd.concat([df_en, df_zh], ignore_index=True)
    # random sample 100 rows, and recount from 0
    # df = df.sample(n=4096, random_state=1)
    df = df.reset_index(drop=True)

    # create a GTMCorpus object
    train_dataset = GTMCorpus(
      df,
      count_words=True,
      embeddings_type='SentenceTransformer',
      sbert_model_to_load=model_name,
      content=None,
      labels='~ category - 1',
      batch_size=batch_size,
      max_seq_length=max_seq_length)

    print('Saving train_dataset_{}-{}.pkl'.format(model_name.replace('/', '_'), 'en-zh'))
    with open('train_dataset_{}-{}.pkl'.format(model_name.replace('/', '_'), 'en-zh'), 'wb') as f:
      p.dump(train_dataset, f)
  else:
    print('train_dataset_{}-{}.pkl already exists'.format(model_name.replace('/', '_'), 'en-zh'))
    with open('train_dataset_{}-{}.pkl'.format(model_name.replace('/', '_'), 'en-zh'), 'rb') as f:
      train_dataset = p.load(f)
  return train_dataset

def train(lr=0.01, w_pred_loss=1.0, encode_bs=256, train_bs=256, model_name='sentence-transformers/all-mpnet-base-v2',
          predict_language=True, epochs=100, w_lang=1.0):
  train_dataset = create_dataset(batch_size=encode_bs, model_name=model_name)
  tm = GTM(
    train_dataset,
    n_topics=6,
    doc_topic_prior='dirichlet',  # logistic_normal, dirichlet
    alpha=0.02,
    update_prior=False,
    encoder_input='embeddings',  # 'bow', 'embeddings'
    decoder_output='bow',  # 'bow', 'embeddings',
    separate_encoders=False, # both languages share the same encoder
    separate_decoders=False, # both languages share the same decoder
    encoder_hidden_layers=[512],  # structure of the encoder neural net
    decoder_hidden_layers=[256],  # structure of the decoder neural net
    encoder_bias=True,
    decoder_bias=True,
    predictor_type='classifier',
    num_epochs=epochs,
    print_every=30,
    dropout=0.0,
    learning_rate=lr,
    log_every=1,
    w_prior=None,
    batch_size=train_bs,
    patience=5,
    save_path='../ckpt2/task6',
    w_pred_loss=w_pred_loss,
    predict_language=predict_language,
    w_lang=w_lang,
    # ckpt='../ckpt2/task6/best_model.ckpt',
  )
  return tm, train_dataset
def inspect(tm, ds):
  doc_topic_distribution = tm.get_doc_topic_distribution(ds)

  print('Number of documents per topic')
  print('Topic 0: {}'.format((doc_topic_distribution.argmax(-1) == 0).sum()))
  print('Topic 1: {}'.format((doc_topic_distribution.argmax(-1) == 1).sum()))
  print('Topic 2: {}'.format((doc_topic_distribution.argmax(-1) == 2).sum()))
  print('Topic 3: {}'.format((doc_topic_distribution.argmax(-1) == 3).sum()))
  print('Topic 4: {}'.format((doc_topic_distribution.argmax(-1) == 4).sum()))
  print('Topic 5: {}'.format((doc_topic_distribution.argmax(-1) == 5).sum()))

  # show five random documents per topic
  for topic in range(tm.n_topics):
    print('Topic {}'.format(topic))
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    for i in np.random.choice(np.where(doc_topic_distribution.argmax(-1) == topic)[0], 5):
      print('=' * 50)
      print(ds.df.iloc[i]['doc_clean'])
      print('----------')
      print('Topic distribution = {}'.format(doc_topic_distribution[i]))

def write_topic_to_file(topic_id, ds, doc_topic_distribution, path):

  with open(path, 'w') as f:
    for i in np.where(doc_topic_distribution.argmax(-1) == topic_id)[0]:
      f.write(ds.df.iloc[i]['doc_clean'] + '\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train GTM')
  parser.add_argument('--w_pred_loss', type=float, default=1.0, help='Weight of prediction loss')
  parser.add_argument('--model_name', type=str, default='sentence-transformers/all-mpnet-base-v2',
                      help='mixedbread-ai/mxbai-embed-large-v1 or sentence-transformers/all-mpnet-base-v2')
  parser.add_argument('--language', type=str, default='en-zh', help='Language')
  parser.add_argument('--encode_bs', type=int, default=256, help='Batch size for encoding')
  parser.add_argument('--train_bs', type=int, default=256, help='Batch size for training')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
  parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
  parser.add_argument('--predict_language', action='store_true', help='Predict language')
  parser.add_argument('--w_lang', type=float, default=1.0, help='Weight of language prediction loss')
  args = parser.parse_args()
  # print all the arguments
  print(args)
  # create output_dir by concatenating the model_name, language, encode_bs, train_bs, lr, and epochs
  output_dir = '../data/task6/m_{}_l_{}_encode_bs_{}_train_bs_{}_lr_{}_e_{}_p_{}_wl_{}'.format(
    args.model_name, args.language, args.encode_bs, args.train_bs, args.lr, args.epochs, args.predict_language,
    args.w_lang)

  output_dir.replace('/', '_')
  print(output_dir)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  tm, train_dataset = train(lr=args.lr, encode_bs=args.encode_bs, train_bs=args.train_bs, w_pred_loss=args.w_pred_loss,
                            model_name=args.model_name, epochs=args.epochs, predict_language=args.predict_language,
                            w_lang=args.w_lang)
  doc_topic_distribution = tm.get_doc_topic_distribution(train_dataset)


  write_topic_to_file(0, train_dataset, doc_topic_distribution, os.path.join(output_dir, 'topic_0.txt'))
  write_topic_to_file(1, train_dataset, doc_topic_distribution, os.path.join(output_dir, 'topic_1.txt'))
  write_topic_to_file(2, train_dataset, doc_topic_distribution, os.path.join(output_dir, 'topic_2.txt'))
  write_topic_to_file(3, train_dataset, doc_topic_distribution, os.path.join(output_dir, 'topic_3.txt'))
  write_topic_to_file(4, train_dataset, doc_topic_distribution, os.path.join(output_dir, 'topic_4.txt'))
  write_topic_to_file(5, train_dataset, doc_topic_distribution, os.path.join(output_dir, 'topic_5.txt'))

  def read_labels(language='en'):
    with open('../data/wiki_shorts/{}/corpus/docs.txt'.format(language), 'r') as file:
      docs = file.readlines()
    with open('../data/wiki_shorts/{}/labels.txt'.format(language), 'r') as file:
      labels = file.readlines()

    # one to one mapping of docs to labels
    doc2label = {}
    for i in range(len(docs)):
      doc2label[docs[i][:100].strip()] = int(labels[i].strip())
    return doc2label

  doc2label_en = read_labels('en')
  doc2label_zh = read_labels('zh')

  # for each topic, find its majority label
  from collections import defaultdict

  def find_majority_label(topic):
    label2cnt = defaultdict(int)
    labels = []
    with open(os.path.join(output_dir, 'topic_{}.txt'.format(topic)), 'r') as file:
      lines = file.readlines()
      for line in lines:
        k = line[:100].strip()
        if k in doc2label_en:
          v = doc2label_en[k]
        else:
          v = doc2label_zh[k]
        label2cnt[v] += 1
        labels.append(v)
    predicted = max(label2cnt, key=lambda k: label2cnt[k])
    return label2cnt, labels, predicted

  _, labels_0, predicted_0 = find_majority_label(0)
  _, labels_1, predicted_1 = find_majority_label(1)
  _, labels_2, predicted_2 = find_majority_label(2)
  _, labels_3, predicted_3 = find_majority_label(3)
  _, labels_4, predicted_4 = find_majority_label(4)
  _, labels_5, predicted_5 = find_majority_label(5)

  print(predicted_0, predicted_1, predicted_2, predicted_3, predicted_4, predicted_5)
  final_labels = labels_0 + labels_1 + labels_2 + labels_3 + labels_4 + labels_5
  final_pred = [predicted_0] * len(labels_0) + [predicted_1] * len(labels_1) + [predicted_2] * len(labels_2) + [
    predicted_3] * len(labels_3) + [predicted_4] * len(labels_4) + [predicted_5] * len(labels_5)

  model_pred = [0] * len(labels_0) + [1] * len(labels_1) + [2] * len(labels_2) + [3] * len(labels_3) + [4] * len(
    labels_4) + [5] * len(labels_5)

  from sklearn.metrics import f1_score, accuracy_score, adjusted_rand_score

  f1_macro = f1_score(y_true=final_labels, y_pred=final_pred, average='macro')
  f1_micro = f1_score(y_true=final_labels, y_pred=final_pred, average='micro')
  acc = accuracy_score(y_true=final_labels, y_pred=final_pred)
  ars = adjusted_rand_score(labels_true=final_labels, labels_pred=model_pred)

  print(args)
  # use args to form a file name for the results
  file_name = os.path.join(output_dir, 'results.txt')
  with open(file_name, 'w') as file:
    file.write('f1_macro = {}\n'.format(f1_macro))
    file.write('f1_micro = {}\n'.format(f1_micro))
    file.write('acc = {}\n'.format(acc))
    file.write('ars = {}\n'.format(ars))

  print('f1_macro = {}'.format(f1_macro))
  print('f1_micro = {}'.format(f1_micro))
  print('acc = {}'.format(acc))
  print('ars = {}'.format(ars))
