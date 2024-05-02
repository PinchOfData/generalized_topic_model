import pandas as pd
import sys

from gtm.corpus_task8 import GTMCorpus
from gtm.gtm_customized import GTM
import pickle as p
import argparse
import numpy as np
import os
from corpus import GTMCorpus as GTMCorpusOld

language = 'en-zh'

def load_examples(path='./data/wiki_shorts/docs_merged.txt', language='en'):
  # read the documents
  df = pd.read_csv(path, header=None, delimiter='\t', names=['label', 'doc_clean'])
  # add a language column
  df['language'] = language
  return df

def merge_df(df_original, df_translated):
  # add a new column of doc_clean_translated to df_original
  df_original['doc_clean_translated'] = df_translated['doc_clean']
  # add a new column of doc_clean_translated to df_translated
  df_translated['doc_clean_translated'] = df_original['doc_clean']
  return df_original, df_translated

# try mixedbread-ai/mxbai-embed-large-v1 or sentence-transformers/all-mpnet-base-v2
def create_dataset(model_name='sentence-transformers/all-mpnet-base-v2',
                   batch_size=64,
                   max_seq_length=512,
                   load_path_en='./data/wiki_shorts/en/corpus/docs_original.txt',
                   load_path_en2zh='./data/wiki_shorts/en/corpus/docs_translated_cut.txt',
                   load_path_zh='./data/wiki_shorts/zh/corpus/docs_original.txt',
                   load_path_zh2en='./data/wiki_shorts/zh/corpus/docs_translated.txt'):
  save_name = 'train_dataset_t8_{}.pkl'.format(model_name.replace('/', '_'))
  if not os.path.exists(save_name):
    df_en = load_examples(load_path_en, 'en')
    df_en2zh = load_examples(load_path_en2zh, 'zh')
    df_zh = load_examples(load_path_zh, 'zh')
    df_zh2en = load_examples(load_path_zh2en, 'en')

    df_en, _ = merge_df(df_en, df_en2zh)
    df_zh, _ = merge_df(df_zh, df_zh2en)

    df = pd.concat([df_en, df_zh], ignore_index=True)

    # # random sample 100 rows, and recount from 0
    # df = df.sample(n=64, random_state=1)
    # df = df.reset_index(drop=True)

    # create a GTMCorpus object
    train_dataset = GTMCorpus(
      df,
      count_words=True,
      embeddings_type='SentenceTransformer',
      sbert_model_to_load=model_name,
      content=None,
      labels='label',
      batch_size=batch_size,
      max_seq_length=max_seq_length)

    print('Saving {}'.format(save_name))
    with open(save_name, 'wb') as f:
      p.dump(train_dataset, f)
  else:
    print('Loading {}'.format(save_name))
    with open(save_name, 'rb') as f:
      train_dataset = p.load(f)
  return train_dataset

def train(lr=0.01, w_pred_loss=1.0, encode_bs=256, train_bs=256, model_name='sentence-transformers/all-mpnet-base-v2',
          predict_language=True, epochs=100, w_lang=1.0, load_path='./data/wiki_shorts/docs_merged.txt',
          encoder_input='embeddings', decoder_output='bow',
          separate_encoder=False, separate_decoder=False, log_file_name=None, summary_dir=None, ckpt_dir=None,
          dropout=0.0, w_prior=None, encoder_hidden_layers=[], decoder_hidden_layers=[256], predictor_hidden_layers=[],
          ckpt_path=None):
  train_dataset2 = create_dataset(batch_size=encode_bs, model_name=model_name)
  train_dataset2.labels = None
  # with open('train_dataset_intfloat-e5-large2-{}.pkl'.format('en-zh'), 'rb') as f:
  #   train_dataset = p.load(f)
  # ckpt_path = summary_dir.replace('summary', 'ckpt') + '/best_model.ckpt'
  # ckpt_path = ckpt_path.replace('e_0', 'e_500')
  tm = GTM(
    train_dataset2,
    n_topics=6,
    doc_topic_prior='dirichlet',  # logistic_normal, dirichlet
    alpha=0.02,
    update_prior=False,
    encoder_input=encoder_input,  # 'bow', 'embeddings'
    decoder_output=decoder_output,  # 'bow', 'embeddings',
    encoder_non_linear_activation='relu',
    decoder_non_linear_activation='relu',
    predictor_non_linear_activation='relu',
    separate_encoders=separate_encoder,
    separate_decoders=separate_decoder,
    encoder_hidden_layers=encoder_hidden_layers,  # structure of the encoder neural net
    decoder_hidden_layers=decoder_hidden_layers,  # structure of the decoder neural net
    predictor_hidden_layers=predictor_hidden_layers,  # structure of the predictor neural net
    encoder_bias=True,
    decoder_bias=True,
    predictor_bias=True,
    predictor_type='classifier',
    num_epochs=epochs,
    print_every=1,
    dropout=dropout,
    learning_rate=lr,
    log_every=1,
    w_prior=w_prior,
    batch_size=train_bs,
    patience=20000,
    save_path=ckpt_dir,
    w_pred_loss=w_pred_loss,
    predict_language=predict_language,
    w_lang=w_lang,
    log_file_name=log_file_name,
    summary_dir=summary_dir,
  )
  # return the SummaryWriter object
  return tm, train_dataset2, tm.writer

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
                      help='mixedbread-ai/mxbai-embed-large-v1 or sentence-transformers/all-mpnet-base-v2 '
                           'or intfloat/multilingual-e5-large or intfloat/multilingual-e5-large-instruct')
  parser.add_argument('--load_path', type=str, default='./data/wiki_shorts/docs_merged.txt', help='Path to the data')
  parser.add_argument('--language', type=str, default='en-zh', help='Language')
  parser.add_argument('--encode_bs', type=int, default=256, help='Batch size for encoding')
  parser.add_argument('--train_bs', type=int, default=256, help='Batch size for training')
  parser.add_argument('--separate_encoder', action='store_true', help='Separate encoder')
  parser.add_argument('--separate_decoder', action='store_true', help='Separate decoder')
  parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
  parser.add_argument('--encoder_input', type=str, default='embeddings', help='Encoder input')
  parser.add_argument('--decoder_output', type=str, default='bow', help='Decoder output')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
  parser.add_argument('--predict_language', action='store_true', help='Predict language')
  parser.add_argument('--w_lang', type=float, default=1.0, help='Weight of language prediction loss')
  parser.add_argument('--w_mmd', type=float, default=1.0, help='Weight of prior loss')
  parser.add_argument('--encoder_hidden_layers', type=int, nargs='+', default=[0], help='Encoder hidden layers')
  parser.add_argument('--decoder_hidden_layers', type=int, nargs='+', default=[0], help='Decoder hidden layers')
  parser.add_argument('--predictor_hidden_layers', type=int, nargs='+', default=[0], help='Predictor hidden layers')
  parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the checkpoint')
  args = parser.parse_args()

  if args.encoder_hidden_layers == [0]:
    args.encoder_hidden_layers = []
  if args.decoder_hidden_layers == [0]:
    args.decoder_hidden_layers = []
  if args.predictor_hidden_layers == [0]:
    args.predictor_hidden_layers = []

  if args.w_mmd == 999.0:
    args.w_mmd = None
  # print all the arguments
  print(args)
  # create output_dir by concatenating the model_name, language, encode_bs, train_bs, lr, and epochs
  output_dir = ('./result/task8/m_{}_l_{}_ebs_{}_tbs_{}_lr_{}_d_{}_e_{}_p_{}_wl_{}_wp_{}_wm_{}'
                '_se_{}_sd_{}_ei_{}_do_{}').format(
    args.model_name.replace('/', '_'), args.language, args.encode_bs, args.train_bs, args.lr, args.dropout,
    args.epochs, args.predict_language, args.w_lang, args.w_pred_loss, args.w_mmd, args.separate_encoder,
    args.separate_decoder, args.encoder_input, args.decoder_output)

  output_dir += '_ehl_{}'.format('_'.join(map(str, args.encoder_hidden_layers)))
  output_dir += '_dhl_{}'.format('_'.join(map(str, args.decoder_hidden_layers)))
  output_dir += '_phl_{}'.format('_'.join(map(str, args.predictor_hidden_layers)))

  print(output_dir)
  summary_dir = os.path.join(output_dir, 'summary')
  ckpt_dir = os.path.join(output_dir, 'ckpt')

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  log_file_name = os.path.join(output_dir, 'log.txt')
  tm, train_dataset, writer = train(lr=args.lr, encode_bs=args.encode_bs, train_bs=args.train_bs, w_pred_loss=args.w_pred_loss,
                            model_name=args.model_name, epochs=args.epochs, predict_language=args.predict_language,
                            w_lang=args.w_lang, load_path=args.load_path,
                            separate_decoder=args.separate_decoder, separate_encoder=args.separate_encoder,
                            encoder_input=args.encoder_input, decoder_output=args.decoder_output,
                            log_file_name=log_file_name, summary_dir=summary_dir, ckpt_dir=ckpt_dir,
                            dropout=args.dropout, w_prior=args.w_mmd, encoder_hidden_layers=args.encoder_hidden_layers,
                            decoder_hidden_layers=args.decoder_hidden_layers,
                            predictor_hidden_layers=args.predictor_hidden_layers,
                            ckpt_path=args.ckpt_path)

  doc_topic_distribution = tm.get_doc_topic_distribution(train_dataset)

  write_topic_to_file(0, train_dataset, doc_topic_distribution, os.path.join(output_dir, 'topic_0.txt'))
  write_topic_to_file(1, train_dataset, doc_topic_distribution, os.path.join(output_dir, 'topic_1.txt'))
  write_topic_to_file(2, train_dataset, doc_topic_distribution, os.path.join(output_dir, 'topic_2.txt'))
  write_topic_to_file(3, train_dataset, doc_topic_distribution, os.path.join(output_dir, 'topic_3.txt'))
  write_topic_to_file(4, train_dataset, doc_topic_distribution, os.path.join(output_dir, 'topic_4.txt'))
  write_topic_to_file(5, train_dataset, doc_topic_distribution, os.path.join(output_dir, 'topic_5.txt'))

  def read_labels():
    # with open('./data/wiki_shorts/docs_merged.txt'.format(language), 'r') as file:
    #   docs = file.readlines()
    # with open('./data/wiki_shorts/labels_merged.txt'.format(language), 'r') as file:
    #   labels = file.readlines()

    with open('./data/wiki_shorts/docs_merged.txt', 'r') as file:
      docs = file.readlines()
    with open('./data/wiki_shorts/labels_merged.txt', 'r') as file:
      labels = file.readlines()

    # one to one mapping of docs to labels
    doc2label = {}
    for i in range(len(docs)):
      _, _, text = docs[i].split('\t')
      doc2label[text[:100].strip()] = int(labels[i].strip())
    return doc2label

  doc2label = read_labels()

  # for each topic, find its majority label
  from collections import defaultdict

  def find_majority_label(topic):
    label2cnt = defaultdict(int)
    # each label has at least one count

    labels = []
    with open(os.path.join(output_dir, 'topic_{}.txt'.format(topic)), 'r') as file:
      lines = file.readlines()
      for line in lines:
        k = line[:100].strip()
        if k in doc2label:
          v = doc2label[k]
        else:
          print('Not found: {}'.format(k))
          # select a random label from 0 to 5
          v = np.random.randint(0, 6)
        label2cnt[v] += 1
        labels.append(v)
    try:
      predicted = max(label2cnt, key=lambda k: label2cnt[k])
    except:
      print('cannot find majority label because of empty topic')
      predicted = np.random.randint(0, 6)
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
    # write all the arguments
    for arg in vars(args):
      file.write('{} = {}\n'.format(arg, getattr(args, arg)))
    file.write('f1_macro = {}\n'.format(f1_macro))
    file.write('f1_micro = {}\n'.format(f1_micro))
    file.write('acc = {}\n'.format(acc))
    file.write('ars = {}\n'.format(ars))

  print('f1_macro = {}'.format(f1_macro))
  print('f1_micro = {}'.format(f1_micro))
  print('acc = {}'.format(acc))
  print('ars = {}'.format(ars))

  # write to tensorboard
  writer.add_scalar('f1_macro', f1_macro, 0)
  writer.add_scalar('f1_micro', f1_micro, 0)
  writer.add_scalar('acc', acc, 0)
  writer.add_scalar('ars', ars, 0)

'''
python gtm_wiki_task8.py --train_bs 512 --epochs 20000 --w_pred_loss 100.0 --encoder_input embeddings --decoder_output embeddings --lr 0.001 --model_name intfloat/multilingual-e5-large --load_path ./data/wiki_shorts/docs_merged.txt

'''