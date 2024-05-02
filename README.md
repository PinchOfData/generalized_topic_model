Torch implementation of the generalized topic model.

Implementation using aligned docs from Google Translate.

First download dataset from https://drive.google.com/file/d/1X0EDDpR3xrSVmrf7M0elvxyl4VR5dMYJ/view?usp=sharing, then unzip the downloaded data.zip in the root directory of the project.

Example Usage:

    python gtm_wiki_task8.py --train_bs 512 --epochs 10000 --w_pred_loss 0.0 --w_mmd 1.0 --encoder_input embeddings --decoder_output bow --separate_encoder --separate_decoder --lr 0.001 --model_name intfloat/multilingual-e5-large --encoder_hidden_layers 256 --decoder_hidden_layers 256 --predictor_hidden_layers 128
