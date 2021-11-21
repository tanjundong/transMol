from tokenizer import SmilesTokenizer
tokenizer = SmilesTokenizer.load('./a.vocab')
gpus = [0,1,]
#gpus = [0,1,2,3,]
#gpus = [4,5,6,7,]
configs = {
    'hidden_dim': 768,
    'ff_dim': 1024,
    'max_len': 80,
    'vocab_size': 100,
    'n_heads': 8,
    'n_encode_layers': 4,
    'n_decode_layers': 4,
    'max_kl_weights' : 2.0,
    'max_epoch': 300,
    'decoder': 'GPT',
    'decode_from_latent': True,
    'predict_adj': True,
    'batch_size': 16*16*int(len(gpus)),
}
configs['vocab_size'] = tokenizer.size


