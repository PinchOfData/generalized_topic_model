from glob import glob
from tqdm import tqdm
import pandas as pd 
import csv
import sys

i = int(sys.argv[1])

if i == 1:
    begin = 0
    end = 999
else:
    begin = (i-1)*1000
    end = i*1000 - 1

filenames = glob('../data/wall_street_journal/*.gz')   
filenames = sorted(filenames)

with open('../data/wall_street_journal/WSJ_articles_{}_{}.csv'.format(begin,end), "w", newline="") as csvfile:
    fieldnames = ["GOID","Date","doc"]
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for file in tqdm(filenames[begin:end]):
        try:
            df = pd.read_csv(file, compression='gzip')
            list_of_dicts = df.to_dict('records')
            for d in list_of_dicts:
                doc = []
                for key, value in d.items():
                    if key not in ['GOID', 'Date']:
                        if value > 0:
                            for i in range(value):
                                doc.append(key)
                d['doc'] = ' '.join(doc)
                writer.writerow([d['GOID'], d['Date'], d['doc']])
        except:
            print(file)           
