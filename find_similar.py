import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from tqdm import tqdm
import os
# t-sne
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# read text from file line by line
def read_text(file_path):
  all_examples = []
  with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
      all_examples.append(line.strip())
  return all_examples

# encode each example using a pretrained model
def encode_examples(examples, model, tokenizer, batch_size):
  all_embeddings = []
  for i in tqdm(range(0, len(examples), batch_size)):
    batch = examples[i:i+batch_size]
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to('mps')
    with torch.no_grad():
      outputs = model(**inputs)
    all_embeddings.append(outputs.pooler_output)
  return torch.cat(all_embeddings)

def process_file(file_path, model, tokenizer, batch_size):
  examples = read_text(file_path)
  embeddings = encode_examples(examples, model, tokenizer, batch_size)
  return examples, embeddings

def find_most_similar(examples_query, embeddings_query, examples_target, embeddings_target, top_k=5):
  # calculate dot-product similarity between query and target embeddings
  similarities = torch.mm(embeddings_query, embeddings_target.T)
  # find top k similar examples
  top_k_similar = torch.topk(similarities, top_k, dim=1)
  # extract top k similar examples for each query example
  top_k_similar_examples = [[examples_target[i] for i in top_k_similar.indices[j]] for j in range(len(examples_query))]
  return top_k_similar, top_k_similar_examples

def save_similar_examples(examples, top_k_similar, top_k_similar_examples, file_path):
  with open(file_path, 'w') as file:
    for i in tqdm(range(len(examples))):
      file.write(f'Query: {examples[i]}\n')
      file.write('Similar examples:\n')
      for j in range(len(top_k_similar_examples[i])):
        file.write(f'{top_k_similar.values[i][j]:.4f} - {top_k_similar_examples[i][j]}\n')
      file.write('\n')

def save_similar_dict(examples, top_k_similar, top_k_similar_examples, file_path):
  similar_dict = {}
  for i in range(len(examples)):
    similar_dict[examples[i]] = top_k_similar_examples[i]
  torch.save(similar_dict, file_path)

def read_labels(file_path):
  all_labels = []
  with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
      all_labels.append(int(line.strip()))
  return all_labels

def main():
  model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
  tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
  model.eval()
  # move model to mps
  if torch.backends.mps.is_available():
    device = torch.device('mps')
    model.to(device)

  if not os.path.exists('data/wiki_shorts/encoded'):
    os.makedirs('data/wiki_shorts/encoded')

  # read english if not on disk
  if not os.path.exists('data/wiki_shorts/encoded/en_docs.pt'):
    print('Processing English')
    examples_en, embeddings_en = process_file('./data/wiki_shorts/en/corpus/docs.txt', model, tokenizer, 32)
    # read labels
    labels_en = read_labels('./data/wiki_shorts/en/labels.txt')
    torch.save(examples_en, 'data/wiki_shorts/encoded/en_docs.pt')
    torch.save(embeddings_en, 'data/wiki_shorts/encoded/en_embeddings.pt')
    torch.save(labels_en, 'data/wiki_shorts/encoded/en_labels.pt')
  else:
    print('Loading English')
    examples_en = torch.load('data/wiki_shorts/encoded/en_docs.pt')
    embeddings_en = torch.load('data/wiki_shorts/encoded/en_embeddings.pt')
    labels_en = torch.load('data/wiki_shorts/encoded/en_labels.pt')

  # read chinese
  if not os.path.exists('data/wiki_shorts/encoded/zh_docs.pt'):
    print('Processing Chinese')
    examples_zh, embeddings_zh = process_file('./data/wiki_shorts/zh/corpus/docs.txt', model, tokenizer, 32)
    labels_zh = read_labels('./data/wiki_shorts/zh/labels.txt')
    torch.save(examples_zh, 'data/wiki_shorts/encoded/zh_docs.pt')
    torch.save(embeddings_zh, 'data/wiki_shorts/encoded/zh_embeddings.pt')
    torch.save(labels_zh, 'data/wiki_shorts/encoded/zh_labels.pt')
  else:
    print('Loading Chinese')
    examples_zh = torch.load('data/wiki_shorts/encoded/zh_docs.pt')
    embeddings_zh = torch.load('data/wiki_shorts/encoded/zh_embeddings.pt')
    labels_zh = torch.load('data/wiki_shorts/encoded/zh_labels.pt')


  # find similar examples
  print('Finding similar examples for English to Chinese...')
  # top_k_similar_for_en, top_k_similar_examples_for_en = find_most_similar(examples_en, embeddings_en, examples_zh, embeddings_zh)
  print('Finding similar examples for Chinese to English...')
  # top_k_similar_for_zh, top_k_similar_examples_for_zh = find_most_similar(examples_zh, embeddings_zh, examples_en, embeddings_en)

  # store on disk
  if not os.path.exists('./data/wiki_shorts/similar'):
    os.makedirs('./data/wiki_shorts/similar')

  print('Saving similar examples to disk...')
  # save_similar_examples(examples_en, top_k_similar_for_en, top_k_similar_examples_for_en, './data/wiki_shorts/similar/en_to_zh.txt')
  # save_similar_examples(examples_zh, top_k_similar_for_zh, top_k_similar_examples_for_zh, './data/wiki_shorts/similar/zh_to_en.txt')

  # store on disk as dictionary
  print('Saving similar examples as dictionary to disk...')
  # save_similar_dict(examples_en, top_k_similar_for_en, top_k_similar_examples_for_en, './data/wiki_shorts/similar/en_to_zh_dict.pt')
  # save_similar_dict(examples_zh, top_k_similar_for_zh, top_k_similar_examples_for_zh, './data/wiki_shorts/similar/zh_to_en_dict.pt')

  # subsample for visualization
  print('Subsampling for visualization...')
  # subsample a random 1000 examples
  indices_en = np.random.choice(len(examples_en), 1000, replace=False)
  indices_zh = np.random.choice(len(examples_zh), 1000, replace=False)
  examples_en = [examples_en[i] for i in indices_en]
  embeddings_en = embeddings_en[indices_en]
  labels_en = [labels_en[i] for i in indices_en]
  examples_zh = [examples_zh[i] for i in indices_zh]
  embeddings_zh = embeddings_zh[indices_zh]
  labels_zh = [labels_zh[i] for i in indices_zh]

  labels_en = np.asarray(labels_en)
  labels_zh = np.asarray(labels_zh)

  # visualize embeddings
  print('Visualizing embeddings...')
  # reduce dimensionality to 3D using PCA, English and Chinese
  pca = PCA(n_components=3)
  embeddings_en_pca = pca.fit_transform(embeddings_en.cpu().numpy())
  embeddings_zh_pca = pca.fit_transform(embeddings_zh.cpu().numpy())
  # plot the 6 topics by labels per language
  fig = plt.figure(figsize=(11, 11))  # Example: set figure size to 10 inches wide by 7 inches tall
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(embeddings_en_pca[:,0], embeddings_en_pca[:,1], embeddings_en_pca[:,2], c=labels_en, cmap='Accent', label='English', marker='o', s=100)
  ax.scatter(embeddings_zh_pca[:,0], embeddings_zh_pca[:,1], embeddings_zh_pca[:,2], c=labels_zh, cmap='Accent', label='Chinese', marker='^', s=100)
  plt.title('PCA, English and Chinese')
  plt.legend()
  plt.tight_layout()  # This line adjusts the padding
  plt.show()

  # reduce dimensionality to 3D using PCA, English only
  pca = PCA(n_components=3)
  embeddings_en_pca = pca.fit_transform(embeddings_en.cpu().numpy())
  # plot the 6 topics by labels per language
  fig = plt.figure(figsize=(11, 11))  # Example: set figure size to 10 inches wide by 7 inches tall
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(embeddings_en_pca[:,0], embeddings_en_pca[:,1], embeddings_en_pca[:,2], c=labels_en, cmap='Accent', label='English', marker='o', s=100)
  plt.title('PCA, English only')
  plt.legend()
  plt.tight_layout()  # This line adjusts the padding
  plt.show()

  # reduce dimensionality to 3D using PCA, Chinese only
  pca = PCA(n_components=3)
  embeddings_zh_pca = pca.fit_transform(embeddings_zh.cpu().numpy())
  # plot the 6 topics by labels per language
  fig = plt.figure(figsize=(11, 11))  # Example: set figure size to 10 inches wide by 7 inches tall
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(embeddings_zh_pca[:,0], embeddings_zh_pca[:,1], embeddings_zh_pca[:,2], c=labels_zh, cmap='Accent', label='Chinese', marker='^', s=100)
  plt.title('PCA, Chinese only')
  plt.legend()
  plt.tight_layout()  # This line adjusts the padding
  plt.show()

  # reduce dimensionality to 3D using t-SNE, English and Chinese
  tsne = TSNE(n_components=3)
  embeddings_en_tsne = tsne.fit_transform(embeddings_en.cpu().numpy())
  embeddings_zh_tsne = tsne.fit_transform(embeddings_zh.cpu().numpy())
  # plot the 6 topics by labels per language
  fig = plt.figure(figsize=(11, 11))  # Example: set figure size to 10 inches wide by 7 inches tall
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(embeddings_en_tsne[:,0], embeddings_en_tsne[:,1], embeddings_en_tsne[:,2], c=labels_en, cmap='Accent', label='English', marker='o', s=100)
  ax.scatter(embeddings_zh_tsne[:,0], embeddings_zh_tsne[:,1], embeddings_zh_tsne[:,2], c=labels_zh, cmap='Accent', label='Chinese', marker='^', s=100)
  plt.title('t-SNE, English and Chinese')
  plt.legend()
  plt.tight_layout()  # This line adjusts the padding
  plt.show()

  # reduce dimensionality to 3D using t-SNE, English only
  tsne = TSNE(n_components=3)
  embeddings_en_tsne = tsne.fit_transform(embeddings_en.cpu().numpy())
  # plot the 6 topics by labels per language
  fig = plt.figure(figsize=(11, 11))  # Example: set figure size to 10 inches wide by 7 inches tall
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(embeddings_en_tsne[:,0], embeddings_en_tsne[:,1], embeddings_en_tsne[:,2], c=labels_en, cmap='Accent', label='English', marker='o', s=100)
  plt.title('t-SNE, English only')
  plt.legend()
  plt.tight_layout()  # This line adjusts the padding
  plt.show()

  # reduce dimensionality to 3D using t-SNE, Chinese only
  tsne = TSNE(n_components=3)
  embeddings_zh_tsne = tsne.fit_transform(embeddings_zh.cpu().numpy())
  # plot the 6 topics by labels per language
  fig = plt.figure(figsize=(11, 11))  # Example: set figure size to 10 inches wide by 7 inches tall
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(embeddings_zh_tsne[:,0], embeddings_zh_tsne[:,1], embeddings_zh_tsne[:,2], c=labels_zh, cmap='Accent', label='Chinese', marker='^', s=100)
  plt.title('t-SNE, Chinese only')
  plt.legend()
  plt.tight_layout()  # This line adjusts the padding
  plt.show()

  # for each topic, visualize the distribution of embeddings, English and Chinese
  print('Visualizing topic distributions...')
  # reduce dimensionality to 3D using PCA, English and Chinese
  pca = PCA(n_components=3)
  embeddings_en_pca = pca.fit_transform(embeddings_en.cpu().numpy())
  embeddings_zh_pca = pca.fit_transform(embeddings_zh.cpu().numpy())
  # plot the 6 topics one by one, by labels per language
  for i in range(6):
    fig = plt.figure(figsize=(11, 11))  # Example: set figure size to 10 inches wide by 7 inches tall
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeddings_en_pca[labels_en==i,0], embeddings_en_pca[labels_en==i,1], embeddings_en_pca[labels_en==i,2], cmap='Accent', label='English', marker='o', s=100)
    ax.scatter(embeddings_zh_pca[labels_zh==i,0], embeddings_zh_pca[labels_zh==i,1], embeddings_zh_pca[labels_zh==i,2], cmap='Accent', label='Chinese', marker='^', s=100)
    plt.title(f'PCA, Topic {i}')
    plt.legend()
    plt.tight_layout()  # This line adjusts the padding
    plt.show()

  # reduce dimensionality to 3D using t-SNE, English and Chinese
  tsne = TSNE(n_components=3)
  embeddings_en_tsne = tsne.fit_transform(embeddings_en.cpu().numpy())
  embeddings_zh_tsne = tsne.fit_transform(embeddings_zh.cpu().numpy())
  # plot the 6 topics one by one, by labels per language
  for i in range(6):
    fig = plt.figure(figsize=(11, 11))  # Example: set figure size to 10 inches wide by 7 inches tall
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeddings_en_tsne[labels_en==i,0], embeddings_en_tsne[labels_en==i,1], embeddings_en_tsne[labels_en==i,2], cmap='Accent', label='English', marker='o', s=100)
    ax.scatter(embeddings_zh_tsne[labels_zh==i,0], embeddings_zh_tsne[labels_zh==i,1], embeddings_zh_tsne[labels_zh==i,2], cmap='Accent', label='Chinese', marker='^', s=100)
    plt.title(f't-SNE, Topic {i}')
    plt.legend()
    plt.tight_layout()  # This line adjusts the padding
    plt.show()

if __name__ == "__main__":
  main()
