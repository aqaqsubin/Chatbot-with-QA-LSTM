
# -*- coding: utf-8 -*-
import os
import pickle
import torch
import transformers
transformers.logging.set_verbosity_error()

import pandas as pd
from torch import nn
from math import ceil as ceil

from os.path import join as pjoin

from data_utils import read_df
from dataloader import ChatTestData
from lightning_model import LightningQALSTM
from model.qa_lstm import RetrievalLSTM
from kobert_transformers import get_tokenizer

TOKENIZER = get_tokenizer()

def base_setting(args):
    args.max_len = getattr(args, 'max_len', 128)
    args.log = getattr(args, 'log', True)
    args.neg_size = getattr(args, 'neg_size', 10)
    args.margin = getattr(args, 'margin', 0.2)
    args.algorithm = getattr(args, 'algorithm', 'levenshtein')
    args.cand_num = getattr(args, 'cand_num', 20)

def load_model(args):
    model = LightningQALSTM(args, tokenizer=TOKENIZER)
    if args.cuda:
        model = model.cuda()

    assert args.model_pt is not None
    model = LightningQALSTM.load_from_checkpoint(checkpoint_path=args.model_pt, hparams=args, tokenizer=TOKENIZER)

    if args.cuda:
        model = model.cuda() 
    return model

def load_reply_embs(model, replies, device):
    with torch.no_grad():
        with open(REACTION_PATH, 'wb') as reaction_file: 
            for reply in replies:
                reply_tok = encode(reply).to(device=device)
                pool, lstm_out = model.get_emb(reply_tok)
                pickle.dump((pool, lstm_out, reply), reaction_file)
                del pool, lstm_out
    return

def get_reply_embs():
    assert os.path.isfile(REACTION_PATH)
    reply_embs = []
    with open(REACTION_PATH, 'rb') as f_react:
        while True:
            try:
                pool, lstm_out, reply = pickle.load(f_react)
            except EOFError:
                break
            reply_embs.append((pool, lstm_out, reply))

    return reply_embs

def tokenize(sent, max_len=128):
    tokens = TOKENIZER.tokenize(TOKENIZER.cls_token + str(sent) + TOKENIZER.sep_token)
    seq_len = len(tokens)
    if seq_len > max_len:
        tokens = tokens[:max_len-1] + tokens[-1]
        seq_len = len(tokens)
        assert seq_len == len(tokens), f'{seq_len} ==? {len(tokens)}'
        
    token_ids = TOKENIZER.convert_tokens_to_ids(tokens)
    while len(token_ids) < max_len:
        token_ids += [TOKENIZER.pad_token_id]

    return token_ids

def encode(sent):
    tok_ids = tokenize(sent)
    return torch.unsqueeze(torch.LongTensor(tok_ids), 0)

def chat(model, cand_num, device, use_attention=False):
    def _is_valid(query):
        if not query or query == "c":
            return False
        return True

    dataset = ChatTestData(data_path=REACTION_PATH, batch_size=1000)
    replies = get_reply_embs()
    query = input("사용자 입력> ")
    
    while _is_valid(query):
        print(f"Query    : {query}")
        query_tok = encode(query).to(device=device)
        query_pool, _ = model.get_emb(query_tok)

        entire_cos_sim = []
        cos_sims = []

        for replies in dataset:
            if not use_attention:
                print("RetrievalLSTM")
                cos_sims = list(map(lambda x: (model.get_similarity(query_pool, x[0].to(device=device)), x[-1]), replies)) 
            else:
                print("RetrievalABLSTM")
                attn_weights = list(map(lambda x: model.get_attn(query_pool, x[1]), replies))
                cos_sims = list(map(lambda x: (model.get_similarity(query_pool, x[0].to(device=device)), x[-1][-1]), zip(attn_weights, replies)))    
                del attn_weights

            cos_sims = sorted(cos_sims, key=lambda x: x[0], reverse=True)
            cos_sims = cos_sims[:cand_num]

            entire_cos_sim += cos_sims
            entire_cos_sim = sorted(entire_cos_sim, key=lambda x: x[0], reverse=True)
            entire_cos_sim = entire_cos_sim[:cand_num]

        candidates = [r for _sim, r in entire_cos_sim]
        print(f"Candidate: {candidates}\n")

        del query_tok, query_pool, _, candidates, entire_cos_sim

        query = input("사용자 입력> ")

    return
        
        
def evaluation(args):
    base_setting(args)
    gpuid = args.gpuid[0]
    device = "cuda:%d" % gpuid

    model = load_model(args)
    model.eval()

    model = model.qa_lstm

    global REACTION_PATH
    REACTION_PATH = pjoin(args.data_dir, 'reaction_emb.pickle')

    if not os.path.isfile(REACTION_PATH):
        reaction_db = read_df(pjoin(args.data_dir, 'reaction.csv'))
        reaction_db.dropna(axis=0, inplace=True)
        with torch.no_grad():    
            load_reply_embs(model=model, replies=reaction_db['reply'], device=device)

    chat(model, cand_num=args.cand_num, device=device, use_attention=args.attention)
        