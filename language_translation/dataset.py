import torch 
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds,tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len:int) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor(tokenizer_src.token_to_id('<SOS>'),dtype=torch.int64)
        self.eos_token = torch.tensor(tokenizer_src.token_to_id('<EOS>'),dtype=torch.int64)
        self.pad_token = torch.tensor(tokenizer_src.token_to_id('<PAD>'),dtype=torch.int64)

        self.unk_token = torch.tensor(tokenizer_src.token_to_id('<UNK>'),dtype=torch.int64)
        self.mask_token = torch.tensor(tokenizer_src.token_to_id('<MASK>'),dtype=torch.int64)


    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        src_trt_text = self.ds[idx]
        src_text = src_trt_text['translation'][self.src_lang]
        tgt_text = src_trt_text['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # -2 for [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # -1 for [EOS]
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Input sequence is too long for the model. "
                             f"Max length is {self.seq_len}, but got {len(enc_input_tokens)} for source and {len(dec_input_tokens)} for target.")
        
        encoder_input = torch.cat(
            [
                self.sos_token.unsqueeze(0),
                torch.tensor(enc_input_tokens, dtype=torch.int64), 
                self.eos_token.unsqueeze(0),
                # self.pad_token.repeat(enc_num_padding_tokens)
                torch.tensor([self.pad_token]* enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        decoder_input = torch.cat(
            [
                self.sos_token.unsqueeze(0),
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                torch.tensor([self.pad_token]* dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        #target output is decoder input shifted by one token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                self.eos_token.unsqueeze(0),
                torch.tensor([self.pad_token]* dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        assert len(encoder_input) == self.seq_len, f"Encoder input length {len(encoder_input)} does not match seq_len {self.seq_len}"
        assert len(decoder_input) == self.seq_len, f"Decoder input length {len(decoder_input)} does not match seq_len {self.seq_len}"
        assert len(label) == self.seq_len, f"Label length {len(label)} does not match seq_len {self.seq_len}"
        
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # Shape: (1, 1, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # Shape: (1, seq_len) & (1,seq_len, seq_len)
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size, size), diagonal=1).type(torch.int)
    return mask == 0  # Invert the mask to get the causal mask (True for allowed positions)