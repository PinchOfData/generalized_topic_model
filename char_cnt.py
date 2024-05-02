import os

file_name = './data/wiki_shorts/docs_merged.txt'

with open(file_name, 'r') as f:
  lines = f.readlines()
  # count the number of characters in the file
  char_cnt = sum([len(line) for line in lines])
  print(f'Number of characters in the file: {char_cnt}')
