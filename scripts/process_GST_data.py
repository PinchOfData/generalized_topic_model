import pandas as pd 
from glob import glob
from tqdm import tqdm 
import sys
sys.path.append('../gtm/')
from utils import text_processor

dataset = str(sys.argv[1]) # can be 'bound' or 'daily'

"""
Speeches = the US Congressional speech records.
"""

print('Loading speeches...')

filelist = sorted(glob('../data/us_congressional_record/hein-{}/speeches*.txt'.format(dataset)))

col_names = ['speech_id', 'doc_clean']
df = pd.DataFrame(columns = col_names)

for file in filelist: # I read the csvs in this way because pd.read_csv displays errors

    print(file)
	
    with open(file, 'rb') as f:
        lines = f.readlines()

    split_lines = [
        str(line).strip().split("|")
        for line in lines
    ]

    speech_ids, speeches = zip(*split_lines[1:])
    temp = pd.DataFrame({'speech_id': speech_ids, 'doc': speeches})

    print('Processing the text data...')
        
    p = text_processor(
        'en_core_web_sm', 
        pos_tags_to_keep = ['VERB', 'NOUN', 'PNOUN', 'ADJ']
    )

    temp['doc_clean'] = p.process_docs(temp['doc'], batch_size = 10)
    
    temp = temp[['speech_id', 'doc_clean']]
    
    idx = [k for k,v in enumerate(list(temp['doc_clean'])) if len(str(v).split()) >= 20] # drop speeches of less than 20 preprocessed tokens
    temp = temp.iloc[idx]
    
    df = pd.concat([df,temp], ignore_index = True)

df.speech_id = df.speech_id.str.replace('b\'', '') 

"""
Speakermaps = Data that relates speaker information with speech ids.
"""

print('Loading speaker maps...')

filelist = glob('../data/us_congressional_record/hein-{}/*SpeakerMap.txt'.format(dataset)) 
df_list = [pd.read_csv(file, sep = '|') for file in tqdm(filelist)] 
speaker_map = pd.concat(df_list, ignore_index = True) 

speaker_map['speech_id']=speaker_map['speech_id'].apply(str)

df = pd.merge(speaker_map, df, on = 'speech_id') 

del speech_ids
del speeches
del temp
del speaker_map

df = df[df['party'].isin(['R', 'D'])]

df.to_csv('../data/us_congressional_record/us_congress_speeches_{}_processed.csv'.format(dataset))
