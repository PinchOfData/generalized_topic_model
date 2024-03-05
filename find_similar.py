import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from tqdm import tqdm
import os

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

def main():
  model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
  tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
  model.eval()
  # move model to mps
  if torch.backends.mps.is_available():
    device = torch.device('mps')
    model.to(device)

  if not os.path.exists('./data/wiki_shorts/encoded'):
    os.makedirs('./data/wiki_shorts/encoded')

  # read english if not on disk
  if not os.path.exists('./data/wiki_shorts/encoded/en_docs.pt'):
    print('Processing English')
    examples_en, embeddings_en = process_file('./data/wiki_shorts/en/corpus/docs.txt', model, tokenizer, 32)
    torch.save(examples_en, './data/wiki_shorts/encoded/en_docs.pt')
    torch.save(embeddings_en, './data/wiki_shorts/encoded/en_embeddings.pt')
  else:
    print('Loading English')
    examples_en = torch.load('./data/wiki_shorts/encoded/en_docs.pt')
    embeddings_en = torch.load('./data/wiki_shorts/encoded/en_embeddings.pt')

  # read chinese
  if not os.path.exists('./data/wiki_shorts/encoded/zh_docs.pt'):
    print('Processing Chinese')
    examples_zh, embeddings_zh = process_file('./data/wiki_shorts/zh/corpus/docs.txt', model, tokenizer, 32)
    torch.save(examples_zh, './data/wiki_shorts/encoded/zh_docs.pt')
    torch.save(embeddings_zh, './data/wiki_shorts/encoded/zh_embeddings.pt')
  else:
    print('Loading Chinese')
    examples_zh = torch.load('./data/wiki_shorts/encoded/zh_docs.pt')
    embeddings_zh = torch.load('./data/wiki_shorts/encoded/zh_embeddings.pt')

  # find similar examples
  top_k_similar_for_en, top_k_similar_examples_for_en = find_most_similar(examples_en, embeddings_en, examples_zh, embeddings_zh)
  top_k_similar_for_zh, top_k_similar_examples_for_zh = find_most_similar(examples_zh, embeddings_zh, examples_en, embeddings_en)

  # store on disk
  if not os.path.exists('./data/wiki_shorts/similar'):
    os.makedirs('./data/wiki_shorts/similar')

  save_similar_examples(examples_en, top_k_similar_for_en, top_k_similar_examples_for_en, './data/wiki_shorts/similar/en_to_zh.txt')
  save_similar_examples(examples_zh, top_k_similar_for_zh, top_k_similar_examples_for_zh, './data/wiki_shorts/similar/zh_to_en.txt')

if __name__ == "__main__":
  main()
