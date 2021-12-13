from tokenizer import SmilesTokenizer
tokenizer = SmilesTokenizer.load('./a.vocab')
gpus = [0,1,]
#gpus = [0,1,2,3,]
#gpus = [4,5,6,7,]
configs = {
    'hidden_dim': 768,
    'ff_dim': 1024*2,
    'max_len': 80,
    'vocab_size': 100,
    'n_heads': 4,
    'n_encode_layers': 8,
    'n_decode_layers': 4,
    'max_kl_weights' : 0.9,
    'max_kl_epoch': 100,
    'max_epoch': 100,
    'decoder': 'RNN',
    'decode_from_latent': True,
    'predict_adj': True,
    'batch_size': 16*16*2*int(len(gpus)),
    'warmup_steps': 30000,
}
configs['vocab_size'] = tokenizer.size


