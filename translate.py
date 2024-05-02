# translate English documents to Chinese documents or Chinese documents to English documents
import argparse
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel
import torch
from tqdm import tqdm

def translate(input_file, output_file, model_name, batch_size):
  # Load model
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  # device, cuda, mps, or cpu
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # mps has bugs
  model.to(device)

  # Read input file
  with open(input_file, 'r') as f:
    lines = f.readlines()

  cnt = 0
  # Translate
  with open(output_file, 'w') as f:
    for i in tqdm(range(0, len(lines), batch_size)):
      batch = lines[i:i+batch_size]
      # always pad to 512
      inputs = tokenizer(batch, return_tensors='pt', padding='max_length', truncation=True)
      inputs = inputs.to(device)
      outputs = model.generate(**inputs)
      decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
      for line in decoded:
        f.write('{}\t{}\n'.format(cnt, line))
        cnt += 1
  print(f'Translation finished. Output file: {output_file}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Translate English documents to Chinese documents or Chinese documents to English documents')
  parser.add_argument('--src', type=str, default='zh', help='source language')
  parser.add_argument('--tgt', type=str, default='en', help='target language')
  parser.add_argument('--batch_size', type=int, default=4, help='batch size')

  args = parser.parse_args()
  model_name = 'Helsinki-NLP/opus-mt-{}-{}'.format(args.src, args.tgt)
  input = './data/wiki_shorts/{}/corpus/docs.txt'.format(args.src)
  output = './data/wiki_shorts/{}/corpus/docs_{}_translated.txt'.format(args.tgt, args.src)

  translate(input, output, model_name, args.batch_size)
