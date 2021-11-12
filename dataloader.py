from torch.utils.data import DataLoader, Dataset

from data_utils import read_df
from typing import List

import numpy as np
import textdistance
import warnings

warnings.filterwarnings(action='ignore')

PAD_TOK = '[PAD]'
CLS_TOK = '[CLS]'
SEP_TOK = '[SEP]'

class ChatData(Dataset):
    def __init__(self, data_path, cand_data_path, tokenizer, algorithm='levenshtein', neg_size=10, max_len=128):
        self._data = read_df(data_path)
        self.cls = CLS_TOK
        self.sep = SEP_TOK
        self.pad = PAD_TOK
        self.max_len = max_len
        self.algorithm = algorithm
        self.neg_size = neg_size
        self.reactions = read_df(cand_data_path)
        self.reactions.dropna(axis=0, inplace=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def _cal_text_distance(self, reply : str, candidates : List[str], algorithm : str):

        if algorithm == 'hamming':
            scores = list(map(lambda x: (x, 1 - textdistance.hamming.normalized_distance(reply, x)), candidates))
        elif algorithm == 'jaro_winkler':
            scores = list(map(lambda x: (x, 1 - textdistance.jaro_winkler.normalized_distance(reply, x)), candidates))
        elif algorithm == 'levenshtein':
            scores = list(map(lambda x: (x, 1 - textdistance.levenshtein.normalized_distance(reply, x)), candidates))
        else:
            raise Exception("Unknown Algorithm : %s" % algorithm)

        return scores
    
    def get_neg_pools(self, reply : str, candidates : List[str]):
        scores = self._cal_text_distance(reply, candidates, algorithm=self.algorithm)
        scores = sorted(scores, key=lambda x: x[-1], reverse=False)
        return [r for r, score in scores[:self.neg_size]]

    def _tokenize(self, sent):
        tokens = self.tokenizer.tokenize(self.cls + str(sent) + self.sep)
        seq_len = len(tokens)
        if seq_len > self.max_len:
            tokens = tokens[:self.max_len-1] + [tokens[-1]]
            seq_len = len(tokens)
            assert seq_len == len(tokens), f'{seq_len} ==? {len(tokens)}'
        
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        return token_ids

    def __getitem__(self, idx):
        turn = self._data.iloc[idx].to_dict()
        query = turn['query']
        reply = turn['reply']
        negs = self.get_neg_pools(reply=reply, candidates=self.reactions['reply'])
        
        # negs = self.reactions.sample(frac=1)[:self.neg_size].reply.tolist()
        query_ids = self._tokenize(query)
        reply_ids = self._tokenize(reply)

        neg_ids = list(map(lambda x: self._tokenize(x), negs))
        
        return(query_ids, reply_ids, neg_ids)
