import torch
import torch.nn as nn
import math
# Transformer Implementation: Input enbedding implemented 
class InputEmbedding(nn.Module):
    def __init__(self,d_model:int, vocab_size: int):
        '''
        d_model: dimension of vector (512)
        vocab_size:  no of words in curpos
        '''
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        # As suggested by paper page 5 3.4
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len:int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position =  torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) #shape (seq_len, 1) 
        div_term =  torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # shape (d_model/2,)
        # use sin for even position and cos for odd position
        # Apply sin on pe
        pe[:,0::2] = torch.sin(position * div_term)
        # Apply cos 
        pe[:,1::2] = torch.cos(position * div_term)

        # Apply batch dimension so that we can apply to all sentense at a go
        pe =  pe.unsqueeze(0) # (1, seq_len, d_model)
        
        '''
        However, sometimes you need to store constant tensors, such as:
            Positional encodings (pe)
            Masks
            Precomputed constants
        These should:
            Be part of the model state (so they get saved/loaded with state_dict()),
            Be moved to the correct device with .cuda() or .to(device),
            But NOT updated by backpropagation (i.e. not learnable).
        That’s what register_buffer is for.
        '''
        self.register_buffer('pe',pe)

    def forward(self, x):
        # if sequence is shorter than max_len, we simply use a slice of the positional encoding. The rest of the rows are ignored. so x.shape[1]
        x =  x + (self.pe[:, :x.shape[1],:]).require_grad_(False)
        return self.dropout(x)





        


