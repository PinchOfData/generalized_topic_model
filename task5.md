We want to align English and Chinese documents.
So we want a mapping M that tells us for each English document, what is the most similar Chinese document, and vice versa.

e.g., M = {1: 997, ..., 997: 1, ...}

doc 1               doc 997
is in                is in
English          Chinese

-> You can use multilingual document embeddings to build M.

I let you decide if you want to work with a dictionary or a matrix.

Once you have M, the idea is to encode doc 1 but to try to reconstruct doc 997, and to encode doc 997 but try to reconstruct doc 1, etc. etc.

We can encode and decode with the bag of words in each language.


python gtm_wiki_task6.py --lr 0.01 --encode_bs 256 --train_bs 256 