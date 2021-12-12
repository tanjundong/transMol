from tokenizer import SmilesTokenizer
tokenizer = SmilesTokenizer.load('./a.vocab')
#gpus = [0,1,]
#gpus = [4,5]
gpus = [0,1,2,3,4,5,6,7,]
configs = {
    'hidden_dim': 768,
    'ff_dim': 1024*2,
    'max_len': 80,
    'vocab_size': 100,
    'n_heads': 1,
    'n_encode_layers': 12,
    'n_decode_layers': 4,
    'max_kl_weights' : 2.0,
    'max_epoch': 200,
    'decoder': 'RNN',
    'decode_from_latent': True,
    'predict_adj': False,
    'batch_size': 16*16*2*int(len(gpus)),
    'warmup_steps': 10000,
}
configs['vocab_size'] = tokenizer.size


