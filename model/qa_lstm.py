import torch
import torch.nn as nn
from transformers import BertModel

class BertEmbeddings(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = BertModel.from_pretrained("monologg/kobert")
        self.pad_token_id = kwargs['pad_token_id']

    def forward(self, tok_ids):
        attn_mask = torch.zeros_like(tok_ids).type_as(tok_ids)
        attn_mask[tok_ids!=self.pad_token_id] = 1
        output = self.model(tok_ids, attn_mask, torch.zeros(tok_ids.size(), dtype=torch.int).type_as(tok_ids))

        return output['last_hidden_state']

class RetrievalLSTM(nn.Module):
    def __init__(self, args, vocab_size, method='max_pooling'):
        super(RetrievalLSTM, self).__init__()
        if args.embed =='bert':
            self.word_embd = BertEmbeddings(pad_token_id=1)
        else:
            self.word_embd = nn.Embedding(vocab_size, args.embd_size)
        self.shared_lstm = nn.LSTM(args.embd_size, args.hidden_size, batch_first=True, bidirectional=True)
        self.cos = nn.CosineSimilarity(dim=1)
        self.method = method

    def get_pools(self, output):
        if self.method == 'max_pooling':
            output = torch.max(output, 1)[0] 
        elif self.method =='avg_pooling':
            output = torch.mean(output, 1)
        else:
            raise NotImplementedError(f'Not Implemented Operation : {self.method}')
        return output

    def forward(self, query, reply):
        # Embedding
        # (batch_size, seq_len, embd_size)
        query_emb = self.word_embd(query)
        reply_emb = self.word_embd(reply)

        # bi-LSTM
        # (batch_size, seq_len, hidden_size * 2)
        q_out, _q_hidden_stat = self.shared_lstm(query_emb)
        r_out, _r_hidden_stat = self.shared_lstm(reply_emb)

        # Get query representation and reply representation by method
        q_pool = self.get_pools(q_out)
        r_pool = self.get_pools(r_out)
        
        cos_sim = self.cos(q_pool, r_pool)
        return cos_sim, q_pool, r_pool

    def get_emb(self, query):
        # Embedding
        # (batch_size, seq_len, embd_size)
        query_emb = self.word_embd(query)

        # bi-LSTM
        # (batch_size, seq_len, hidden_size * 2)
        q_out, _q_hidden_stat = self.shared_lstm(query_emb)

        # Get query representation and reply representation by method
        q_pool = self.get_pools(q_out)
        return q_pool, q_out

    def get_similarity(self, q_pool, r_pool):
        return self.cos(q_pool, r_pool)