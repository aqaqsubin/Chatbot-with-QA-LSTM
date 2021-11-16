import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import BadanauAttn
from .qa_lstm import RetrievalLSTM

class RetrievalABLSTM(RetrievalLSTM):
    def __init__(self, args, vocab_size, method='max_pooling'):
        super(RetrievalABLSTM, self).__init__(args, vocab_size, method)
        self.attn = BadanauAttn(args.hidden_size)


    def forward(self, query, reply):
        # embedding
        # (batch_size, seq_len, embd_size)
        query_emb = self.word_embd(query)
        reply_emb = self.word_embd(reply)

        # bi-LSTM
        # (batch_size, seq_len, hidden_size * 2)
        q_out, q_hidden_stat = self.shared_lstm(query_emb)
        
        # Get representation of query 
        # (batch_size, hidden_size * 2)
        q_pool = self.get_pools(q_out)

        # bi-LSTM
        # (batch_size, seq_len, hidden_size * 2)
        r_out, _r_hidden_stat = self.shared_lstm(reply_emb)

        # Get attention weights
        # (batch_size, seq_len, hidden_size * 2)
        attn_weights = self.attn(q_pool, r_out) 

        # Get representation of reply 
        # (batch_size, hidden_size * 2)
        r_pool = self.get_pools(attn_weights)

        # (batch_size,)
        cos_sim = self.cos(q_pool, r_pool)
        return cos_sim, q_pool, r_pool

    def get_attn(self, q_pool, r_out):

        attn_weights = self.attn(q_pool, r_out) 
        r_pool = self.get_pools(attn_weights)

        return r_pool
        
