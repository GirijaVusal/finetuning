import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
from datasets import load_dataset
from tokenizers import Tokenizer 
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer 
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path 
from dataset import BilingualDataset

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
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        # trainer = WordLevelTrainer(vocab_size=config['vocab_size'], special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],min_frequency=2)
        # tokenizer.train_from_iterator(ds[lang]['train']['text'], trainer=trainer)
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],min_frequency=2)
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
    ds = load_dataset(config['dataset_name'], config['dataset_config'],split="train")

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



