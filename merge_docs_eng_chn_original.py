import os

en_original = './data/wiki_shorts/en/corpus/docs_original.txt'
zh_original = './data/wiki_shorts/zh/corpus/docs_original.txt'

out_file_path = './data/wiki_shorts/docs_en_zh_original.txt'

en_original_f = open(en_original, 'r')
zh_original_f = open(zh_original, 'r')

out_file = open(out_file_path, 'w')

en_original_lines = en_original_f.readlines()
zh_original_lines = zh_original_f.readlines()

print('len(en_original_lines):', len(en_original_lines))
print('len(zh_original_lines):', len(zh_original_lines))

# write en_original_lines
for i in range(len(en_original_lines)):
  label, line = en_original_lines[i].strip().split('\t')
  label = int(label)
  out_file.write('{}\ten\t{}\n'.format(label, line))

length_en = len(en_original_lines)

# write zh_original_lines
for i in range(len(zh_original_lines)):
  label, line = zh_original_lines[i].strip().split('\t')
  label = int(label)
  out_file.write('{}\tzh\t{}\n'.format(label + length_en, line))

# merge labels
en_label_f = open('./data/wiki_shorts/en/labels.txt', 'r')
zh_label_f = open('./data/wiki_shorts/zh/labels.txt', 'r')

merged_labels = open('./data/wiki_shorts/labels_en_zh_original.txt', 'w')

# two times en labels, then two times zh labels
en_labels = en_label_f.readlines()
zh_labels = zh_label_f.readlines()

for i in range(len(en_labels)):
  merged_labels.write('{}\n'.format(en_labels[i].strip()))

for i in range(len(zh_labels)):
  merged_labels.write('{}\n'.format(zh_labels[i].strip()))