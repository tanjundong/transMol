from tokenizer import SmilesTokenizer
tokenizer = SmilesTokenizer.load('./a.vocab')
gpus=2
configs = {
    'hidden_dim': 768,
    'ff_dim': 512,
    'max_len': 64,
    'vocab_size': 100,
    'n_heads': 4,
    'n_encode_layers': 6,
    'n_decode_layers': 4,
    'batch_size': 16*16*(gpus),
}
configs['vocab_size'] = tokenizer.size


