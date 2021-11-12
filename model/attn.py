import torch
from torch import nn
import torch.nn.functional as F

class BadanauAttn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
        self.W_h = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False) 
        self.W_e = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False) 

        self.w_ms = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        # self.w_ms = nn.Linear(2 * hidden_size, 1)

    def forward(self, query_pool, rep_out):
        '''
            query_pool: (batch_size, seq_len)
            rep_out: (batch_size, seq_len, hidden_size * 2)
        '''
        seq_len = rep_out.size(1)

        # (batch_size, seq_len, hidden_size * 2)
        W_s_h = torch.tanh(self.W_h(rep_out) + self.W_e(query_pool).unsqueeze(1).expand(-1, seq_len, -1))

        # Attention Score
        # (batch_size, seq_len, hidden_size * 2)
        attn_score = self.w_ms(W_s_h)
        
        # Alignment Vector (Attention Distribution)
        # (batch_size, seq_len, hidden_size * 2)
        att_weights = F.softmax(attn_score, dim=1)
        
        return att_weights