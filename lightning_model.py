import argparse
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning.core.lightning import LightningModule
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from dataloader import ChatData
from model.qa_lstm_with_attn import RetrievalABLSTM
from model.qa_lstm import RetrievalLSTM

class LightningQALSTM(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(LightningQALSTM, self).__init__()
        self.hparams = hparams
        self.tok = kwargs['tokenizer']
        if not self.hparams.attention:
            self.qa_lstm = RetrievalLSTM(hparams, vocab_size=self.tok.vocab_size, method=self.hparams.method)
        else:
            self.qa_lstm = RetrievalABLSTM(hparams, vocab_size=self.tok.vocab_size, method=self.hparams.method)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_len',
                            type=int,
                            default=128)
        parser.add_argument('--batch_size',
                            type=int,
                            default=32)
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        parser.add_argument('--neg_size',
                            type=int,
                            default=10)
        parser.add_argument('--margin',
                            type=float,
                            default=0.2)

        parser.add_argument('--algorithm',
                            type=str,
                            default='levenshtein')
        parser.add_argument('--num_layers',
                            type=int,
                            default=6)
        parser.add_argument('--hidden_size',
                            type=int,
                            default=768)
        parser.add_argument('--embd_size',
                            type=int,
                            default=768)
        parser.add_argument('--dropout_rate',
                            type=float,
                            default=0.1)

        return parser

    def forward(self, query, reply):
        cos_sim = self.qa_lstm(query, reply)
        return cos_sim

    def loss_fn(self, pos_sim, neg_sim, margin):
        r"""
            Args:
                pos_sim: cosine similarity between query and real reply
                        (batch size, 1)
                neg_sim: cosine similarity between query and a candidate in negative pool
                        (batch size, 1)
                margin: 0.2 (float).  

            Return:
                : list of losses of which size is (batch size, 1)
        """
        margin = torch.ones_like(pos_sim) * margin

        losses = margin - pos_sim + neg_sim
        losses[losses < 0] = 0

        return losses

    def training_step(self, batch, batch_idx):
        query, pos, negs = batch
        pos_sim = self(query, pos)
    
        losses = []
        for _ in range(self.hparams.neg_size):
            neg = negs[:, np.random.randint(self.hparams.neg_size), :]
            neg_sim = self(query, neg)
            loss = self.loss_fn(pos_sim, neg_sim, self.hparams.margin)
            # if loss.data[0] != 0:
            losses.append(loss)
            #     break
        train_loss = torch.mean(torch.stack(losses, 0))
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        query, pos, negs = batch
        pos_sim = self(query, pos)
        
        losses = []
        for _ in range(self.hparams.neg_size):
            neg = negs[:, np.random.randint(self.hparams.neg_size), :]
            neg_sim = self(query, neg)
            loss = self.loss_fn(pos_sim, neg_sim, self.hparams.margin)
            # if loss.data[0] != 0:
            losses.append(loss)
            #     break
        val_loss = torch.mean(torch.stack(losses, 0))
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_losses = []
        for loss_avg in outputs:
            avg_losses.append(loss_avg)
        self.log('val_loss', torch.stack(avg_losses).mean(), prog_bar=True)
    
    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)

        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        query = [item[0] for item in batch]
        pos = [item[1] for item in batch]
        negs = [item[2] for item in batch]
        return torch.LongTensor(query), torch.LongTensor(pos), torch.LongTensor(negs)

    def train_dataloader(self):
        self.train_set = ChatData(data_path=f"{self.hparams.data_dir}/train.csv", cand_data_path=f"{self.hparams.data_dir}/reaction.csv", tokenizer=self.tok, algorithm=self.hparams.algorithm, neg_size=self.hparams.neg_size, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    def val_dataloader(self):
        self.valid_set = ChatData(data_path=f"{self.hparams.data_dir}/val.csv", cand_data_path=f"{self.hparams.data_dir}/reaction.csv", tokenizer=self.tok, algorithm=self.hparams.algorithm, neg_size=self.hparams.neg_size, max_len=self.hparams.max_len)
        val_dataloader = DataLoader(
            self.valid_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return val_dataloader
    

