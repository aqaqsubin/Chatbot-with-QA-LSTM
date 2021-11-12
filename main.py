import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from kobert_transformers import get_tokenizer
from lightning_qa_lstm import LightningQALSTM
from eval import evaluation

import argparse
import transformers
import datetime

import warnings
import torch
from os.path import join as pjoin

warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

TOKENIZER = get_tokenizer()

ROOT_DIR = os.getcwd()
MODEL_DIR = pjoin(ROOT_DIR, 'model_ckpt')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Retrieval Chatbot based on QA-LSTM')
    parser.add_argument('--train',
                        action='store_true',
                        default=False,
                        help='for training')

    parser.add_argument('--data_dir',
                        type=str,
                        default='../data')

    parser.add_argument("--method", type=str, default="max_pooling")

    parser.add_argument("--cuda", 
                        action='store_true',
                        default=False)

    parser.add_argument("--attention", 
                        action='store_true',
                        default=False)

    parser.add_argument("--gpuid", nargs='+', type=int, required=True)

    today = datetime.datetime.now()
    parser.add_argument("--model_name", type=str, default=f"{today.strftime('%m%d')}_qa-lstm")
    parser.add_argument("--model_pt", type=str, default=f'{MODEL_DIR}/model_last.ckpt')

    parser = LightningQALSTM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = args.data_dir

    print(args.gpuid)

    if args.train:
        with torch.cuda.device(args.gpuid[0]):
            checkpoint_callback = ModelCheckpoint(
                dirpath='model_ckpt',
                filename='{epoch:02d}-{train_loss:.2f}',
                verbose=True,
                save_last=True,
                monitor='train_loss',
                mode='min',
                prefix=f'{args.model_name}'
            )

            model = LightningQALSTM(args, tokenizer=TOKENIZER)
            model.train()
            trainer = Trainer(
                            check_val_every_n_epoch=1, 
                            checkpoint_callback=checkpoint_callback, 
                            flush_logs_every_n_steps=100, 
                            gpus=args.gpuid, 
                            gradient_clip_val=1.0, 
                            log_every_n_steps=50, 
                            logger=True, 
                            max_epochs=args.max_epochs, 
                            num_processes=1)
            
            trainer.fit(model)
            print('best model path {}'.format(checkpoint_callback.best_model_path))

    else:
        with torch.cuda.device(args.gpuid[0]):
            evaluation(args)