import torch
import torch.nn as nn
import math

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


