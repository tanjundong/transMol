from tokenizer import SmilesTokenizer
tokenizer = SmilesTokenizer.load('./a.vocab')
#gpus = [0,1,2,3,]
gpus = [4,5,6,7,]
configs = {
    'hidden_dim': 768,
    'ff_dim': 1024,
    'max_len': 64,
    'vocab_size': 100,
    'n_heads': 4,
    'n_encode_layers': 12,
    'n_decode_layers': 3,
    'batch_size': 16*16*4*(len(gpus)),
}
configs['vocab_size'] = tokenizer.size


