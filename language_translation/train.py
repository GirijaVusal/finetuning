import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader, Dataset
from datasets import load_dataset
from tokenizers import Tokenizer 
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer 
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from pathlib import Path 
from dataset import BilingualDataset
from embedding import build_transformer

from config import get_weights_file_path,get_config
def get_all_sentences(ds, lang):
    """
    Extracts all sentences from the dataset for a given language.
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path =  Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        print(f"Building tokenizer for {lang}...")
        tokenizer = Tokenizer(WordLevel(unk_token='<UNK>'))
        tokenizer.pre_tokenizer = Whitespace()
        # trainer = WordLevelTrainer(vocab_size=config['vocab_size'], special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],min_frequency=2)
        # tokenizer.train_from_iterator(ds[lang]['train']['text'], trainer=trainer)
        trainer = WordLevelTrainer(    special_tokens=['<UNK>', '<PAD>', '<CLS>', '<SEP>', '<MASK>', '<SOS>', '<EOS>'],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        print(f"Loading tokenizer for {lang} from {tokenizer_path}...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    """
    Loads the dataset and returns it.
    """
    # ds = load_dataset(config['dataset_name'], config['dataset_config'],split="train")
    ds = load_dataset(config['dataset_name'], f"{config['source_lang']}-{config['target_lang']}", split="train")


    tokenizer_src = get_or_build_tokenizer(config, ds, config['source_lang'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds, config['target_lang'])

    # split to val ds
    train_size = int(0.9 * len(ds))
    val_ds_size = len(ds) - train_size

    train_ds, val_ds = random_split(ds, [train_size, val_ds_size])

    train_ds =  BilingualDataset(train_ds, tokenizer_src, tokenizer_tgt, config['source_lang'], config['target_lang'], config['seq_len'])
    val_ds = BilingualDataset(val_ds, tokenizer_src, tokenizer_tgt, config['source_lang'], config['target_lang'], config['seq_len'])
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    max_len_src = 0
    max_len_tgt = 0

    for item in ds:
        src_ids = tokenizer_src.encode(item['translation'][config['source_lang']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['target_lang']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length \nsource: {max_len_src},\ntarget: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=config['num_workers'])
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, src_vocab_len, tgt_vocab_len):
    """
    Initializes the model with the given configuration and tokenizers.
    build_transformer(src_vocab_size: int, trg_vocab_size: int, src_seq_len:int, trg_seq_len: int, d_model: int = 512, d_ff: int = 2048, n_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):

    """
    model  = build_transformer(src_vocab_len,tgt_vocab_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model


def train_model(config):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model.to(device)

    #Tensorboard writer
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],eps=1e-9)

    # if training crashes this code will help to resume training from the last checkpoint
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading model from {model_filename}...")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('<PAD>'),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['epochs']):
        model.train() 
        bathch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}") 
        for batch in bathch_iterator:
            
            encoder_input = batch['encoder_input'].to(device) #(batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) #(batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #(batch_size,1,1 seq_len)
            decoder_mask = batch['decoder_mask'].to(device) #(batch_size,1,seq_len, seq_len)

            #run tensors through the model
            encoder_output = model.encode(encoder_input, encoder_mask) #(batch_size, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch_size, seq_len, d_model)
            # decoder_output = model.decode(encoder_output, encoder_mask,decoder_input,decoder_mask) # (batch_size, seq_len, d_model)
            projection_output = model.project(decoder_output) #(batch_size, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) #(batch_size, seq_len)
            # (batch_size, seq_len, tgt_vocab_size) --> (batch_size * seq_len, tgt_vocab_size)
            loss = loss_fn(projection_output.view(-1,tokenizer_tgt.get_vocab_size()), label.view(-1))
            bathch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # log to tensorboard
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.flush()

            #backpropagation
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        # save model after each epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)

if __name__ == "__main__":
    # warnings.filterwarnings("ignore", category=UserWarning, module='torch')
    import warnings
    warnings.filterwarnings("ignore")

    config = get_config()
    train_model(config)
    print("Training completed.")
    print(f"Model weights saved in {config['model_folder']}")
    print(f"Tensorboard logs saved in {config['experiment_name']}")

    



