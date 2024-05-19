import jieba
from tqdm import tqdm

# read docs
path = './data/wiki_shorts/en/corpus/docs_translated.txt'
out_path = './data/wiki_shorts/en/corpus/docs_translated_cut.txt'

with open(path, 'r') as f, open(out_path, 'w') as out_f:
  lines = f.readlines()
  # cut line by line
  for line in tqdm(lines):
    text_original = line.strip().split('\t')[1].strip()
    label = line.strip().split('\t')[0].strip()
    text_cutted = list(jieba.cut(text_original))
    text = ' '.join(text_cutted)
    out_f.write(label.strip() + '\t' + text.strip() + '\n')
