from pathlib import Path

def get_config():
    # seq_len need to check on dataset
    config = {
        "batch_size": 2,
        "num_workers": 2,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "source_lang": "en",
        "target_lang": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_enit_",
        "preload":None,
        "tokenizer_path": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel_enit",
        "dataset_name": "opus_books",
        'epochs': 1,
    }
    return config

def get_weights_file_path(config, epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)
   