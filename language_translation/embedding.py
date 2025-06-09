import torch
import torch.nn as nn
import math
# https://arxiv.org/pdf/1706.03762.pdf

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


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #  Multiplicative factor
        self.bias = nn.Parameter(torch.zeros(1)) # Additive factor

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized_x = (x - mean) / (std + self.eps)
        return self.alpha * normalized_x + self.bias
    

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # First linear layer W1 and bias b1
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) 

    def forward(self, x):
        # Apply ReLU activation function and dropout
        # Then apply second linear layer W2 and bias b2
        # x.shape = (batch_size, seq_len, d_model)
        # Output shape will be (batch_size, seq_len, d_model)
        # Note: dropout is applied after the activation function
        # This is a common practice in transformer models to prevent overfitting
        # The dropout layer is applied to the output of the first linear layer after ReLU activation function
        # and before the second linear layer

           
        # (batch_size, seq_len, d_model) ---> (batch_size, seq_len, d_ff) ---> (batch_size, seq_len, d_model)  
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model: int, n_heads:int = 8, dropout: float= 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads #dk # Dimension of each head

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        # query key and value matrices
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out  = nn.Lienar(d_model, d_model)

    @staticmethod
    def attention(query, key, value, dropout=None, mask=None):
        head_dim = query.shape[-1]  # d_k
        # Calculate attention scores
        # (batch_size, n_heads, seq_len, head_dim) @ (batch_size, n_heads, head_dim, seq_len) --> (batch_size, n_heads, seq_len, seq_len)
        attention_scores  =  query @ key.transpose(-2, -1) / math.sqrt(head_dim)  # Scaled dot-product attention
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)  # Apply mask to attention scores
        # Apply softmax to get attention weights   
        attention_scores =  torch.softmax(attention_scores, dim=-1)  # (batch_size, n_heads, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # Multiply attention weights with value vectors
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, head_dim) --> (batch_size, n_heads, seq_len, head_dim)
        output = attention_scores @ value  # (batch_size, n_heads, seq_len, head_dim)
        return output, attention_scores  # Return both output and attention scores for visualization if needed



    def forward(self, query, key, value, mask=None):

        # query, key, value shape: (batch_size, seq_len, d_model) --> (batch_size, seq_len, n_heads, head_dim)
        # We will reshape them to (batch_size, n_heads, seq_len, head_dim) for multi-head attention
        # This is done to split the d_model into n_heads parts
        # and apply attention to each part separately
        # Then we will concatenate the results and apply a linear layer to get the final output


        # Linear projections
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        #split into multiple heads
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        # (batch_size, seq_len, d_model)--> (batch_size, seq_len, n_heads, head_dim) -->(batch_size, n_heads, seq_len, head_dim)
        query =  query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) 
        key   =  key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        value =  value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1,2)

        # Calculate attention scores
        # (batch_size, n_heads, seq_len, head_dim) @ (batch_size, n_heads, head_dim, seq_len) --> (batch_size, n_heads, seq_len, seq_len)
        attention_output, attention_scores = self.attention(query, key, value, self.dropout, mask)
        
        # Concatenate the heads
        # (batch_size, n_heads, seq_len, head_dim) --> (batch_size, seq_len, n_heads * head_dim) --> (batch_size, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Apply final linear layer
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        output = self.out(attention_output)
        return output #, attention_scores 
    

class ResidualConnection(nn.Module):
    # Add and Norm layer

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer_output):
        # x.shape = (batch_size, seq_len, d_model)
        # sublayer_output.shape = (batch_size, seq_len, d_model)
        # Output shape will be (batch_size, seq_len, d_model)
        # add and norm 

        return x + self.dropout(sublayer_output(self.norm(x)))  # Add the output of the sublayer to the input x
    

class EncoderBlock(nn.model):

    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float = 0.1):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward

        # self.residual_connection1 = ResidualConnection(dropout)
        # self.residual_connection2 = ResidualConnection(dropout)
        
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask=None):

        #first add and norm residual connection
        x =  self.residual_connection[0](x, lambda x: self.self_attention(x,x,x,src_mask))
        #second add and norm of encoder  
        x =  self.residual_connection[1](x, lambda x: self.feed_forward)

        return x  # Output shape will be (batch_size, seq_len, d_model)
    


class EncoderOnly(nn.Module):
    def __init__(self, input_embedding: InputEmbedding, positional_embedding: PositionalEmbedding, encoder_block: EncoderBlock, num_layers: int = 6):
        super().__init__()
        self.input_embedding = input_embedding
        self.positional_embedding = positional_embedding
        self.encoder_blocks = nn.ModuleList([encoder_block for _ in range(num_layers)])

    def forward(self, x, src_mask=None):
        # x.shape = (batch_size, seq_len)
        # Apply input embedding and positional embedding
        x = self.input_embedding(x)
        x = self.positional_embedding(x)

        # Pass through each encoder block
        for block in self.encoder_blocks:
            x = block(x, src_mask)

        return x  # Output shape will be (batch_size, seq_len, d_model)
    
class Encoder(nn.Module):
    def __init__(self,layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)  # Output shape will be (batch_size, seq_len, d_model)



class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float = 0.1):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward

        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # x.shape = (batch_size, seq_len, d_model)
        # src_mask : encoder mask 
        # tgt_mask : decoder mask
        # encoder_output.shape = (batch_size, src_seq_len, d_model)

        # First self-attention layer
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))

        # Second cross-attention layer --> cross-attention meaning that output of encoder is used as key and value
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))

        # Feed-forward layer
        x = self.residual_connection[2](x, lambda x: self.feed_forward(x))

        return x  # Output shape will be (batch_size, seq_len, d_model)

class DecoderOnly(nn.Module):
    def __init__(self, input_embedding: InputEmbedding, positional_embedding: PositionalEmbedding, decoder_block: DecoderBlock, num_layers: int = 6):
        super().__init__()
        self.input_embedding = input_embedding
        self.positional_embedding = positional_embedding
        self.decoder_blocks = nn.ModuleList([decoder_block for _ in range(num_layers)])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # x.shape = (batch_size, seq_len)
        # Apply input embedding and positional embedding
        x = self.input_embedding(x)
        x = self.positional_embedding(x)

        # Pass through each decoder block
        for block in self.decoder_blocks:
            x = block(x, encoder_output, src_mask, tgt_mask)

        return x  # Output shape will be (batch_size, seq_len, d_model)
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)  # Output shape will be (batch_size, seq_len, d_model)

