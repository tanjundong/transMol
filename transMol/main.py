from models import get_model, VAE

from dataset import SmilesDataMudule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from tokenizer import SmilesTokenizer

configs = {
    'hidden_dim': 128,
    'ff_dim': 128,
    'max_len': 80,
    'vocab_size': 100,
    'n_heads': 8,
    'n_encode_layers': 3,
    'n_decode_layers': 2,
    'batch_size': 16*16*12,
}

wandb.init(config=configs)
configs = wandb.config

tokenizer = SmilesTokenizer.load('./a.vocab')

model = get_model('trans', configs)

data_model = SmilesDataMudule(
    tokenizer,
    '../data/train',
    '../data/val',
    configs['batch_size'],
    configs['max_len'])


wandb_logger = WandbLogger()

wandb_logger.watch(model.model)

trainer = pl.Trainer(
    gpus=-1,
    logger=wandb_logger,
    max_epochs=100,
    accelerator='dp',
    log_every_n_steps=2,
    gradient_clip_val=0.25,
)

trainer.fit(model, data_model)

