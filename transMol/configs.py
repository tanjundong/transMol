from tokenizer import SmilesTokenizer
tokenizer = SmilesTokenizer.load('./a.vocab')
gpus = 8
configs = {
    'hidden_dim': 768,
    'ff_dim': 1024,
    'max_len': 64,
    'vocab_size': 100,
    'n_heads': 4,
    'n_encode_layers': 8,
    'n_decode_layers': 8,
    'batch_size': 16*16*(gpus),
}
configs['vocab_size'] = tokenizer.size


