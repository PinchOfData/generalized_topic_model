import numpy as np
import pandas as pd
import csv
import warnings
import os

def write_topic_to_file(topic_id, ds, doc_topic_distribution, path):

  with open(path, 'w') as f:
    for i in np.where(doc_topic_distribution.argmax(-1) == topic_id)[0]:
      f.write(ds.df.iloc[i]['doc_clean'] + '\n')

def compute(tm, train_dataset, output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
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
    total_not_found = 0
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
          # print('Not found: {}'.format(k))
          total_not_found += 1
          # select a random label from 0 to 5
          v = np.random.randint(0, 6)
        label2cnt[v] += 1
        labels.append(v)
    try:
      predicted = max(label2cnt, key=lambda k: label2cnt[k])
    except:
      print('cannot find majority label because of empty topic')
      predicted = np.random.randint(0, 6)
    if total_not_found > 0:
      print('total not found: {}'.format(total_not_found))
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

  # use args to form a file name for the results
  file_name = os.path.join(output_dir, 'results.txt')
  with open(file_name, 'w') as file:
    # write all the arguments
    file.write('f1_macro = {}\n'.format(f1_macro))
    file.write('f1_micro = {}\n'.format(f1_micro))
    file.write('acc = {}\n'.format(acc))
    file.write('ars = {}\n'.format(ars))

  print('f1_macro = {}'.format(f1_macro))
  print('f1_micro = {}'.format(f1_micro))
  print('acc = {}'.format(acc))
  print('ars = {}'.format(ars))
  print('-'*50)
