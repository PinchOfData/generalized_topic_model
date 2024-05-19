from os import environ
from tqdm import tqdm
from google.cloud import translate
import argparse
PROJECT_ID = environ.get("PROJECT_ID", "")
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"

def translate_with_gcp(input_file, output_file, src_file, src, tgt):
  client = translate.TranslationServiceClient()
  with open(input_file, 'r') as f:
    lines = f.readlines()
  cnt = 0
  with open(output_file, 'w') as f, open(src_file, 'w') as f_src:
    for line in tqdm(lines):
      response = client.translate_text(
        parent=PARENT,
        contents=[line.strip()],
        target_language_code=tgt,
      )
      translated = response.translations[0]
      f.write('{}\t{}\n'.format(cnt, translated.translated_text))
      f_src.write('{}\t{}\n'.format(cnt, line.strip()))
      cnt += 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Translate English documents to Chinese documents or Chinese documents to English documents')
  parser.add_argument('--src', type=str, default='zh', help='source language')
  parser.add_argument('--tgt', type=str, default='en', help='target language')
  args = parser.parse_args()

  input_file = './data/wiki_shorts/{}/corpus/docs.txt'.format(args.src)
  output_file = './data/wiki_shorts/{}/corpus/docs_translated.txt'.format(args.src)
  src_file = './data/wiki_shorts/{}/corpus/docs_original.txt'.format(args.src)
  translate_with_gcp(input_file, output_file, src_file, args.src, args.tgt)
  print('translation finished, src file: {}, tgt file: {}'.format(src_file, output_file))

'''
python translate_gcp_gtm.py --src en --tgt zh
python translate_gcp_gtm.py --src zh --tgt en

'''