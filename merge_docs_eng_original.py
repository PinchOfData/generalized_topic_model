import os

en_original = './data/wiki_shorts/en/corpus/docs_original.txt'

out_file_path = './data/wiki_shorts/docs_eng_original.txt'

en_original_f = open(en_original, 'r')

out_file = open(out_file_path, 'w')

en_original_lines = en_original_f.readlines()

print('len(en_original_lines):', len(en_original_lines))

# write en_original_lines
for i in range(len(en_original_lines)):
  label, line = en_original_lines[i].strip().split('\t')
  label = int(label)
  out_file.write('{}\ten\t{}\n'.format(label, line))


# merge labels
en_label_f = open('./data/wiki_shorts/en/labels.txt', 'r')

merged_labels = open('./data/wiki_shorts/labels_eng_original.txt', 'w')

# two times en labels, then two times zh labels
en_labels = en_label_f.readlines()

for i in range(len(en_labels)):
  merged_labels.write('{}\n'.format(en_labels[i].strip()))
