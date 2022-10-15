import pytorch_ligthnig as pl
from pytorch_ligthning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils.preprocessor_toxic import PreprocessorToxic

if __name__ == '__main__':

    dm = PreprocessorToxic()
    model = TrainerMultilabel()

    early_stop_callback = EarlyStopping(monitor = 'val_loss', 
                                        min_delta = 0.00,
                                        patience = 3,
                                        mode = "min")
    
    logger = TensorBoardLogger("logs", name="bert_multilabel")

    trainer = pl.Trainer(gpus = 1,
                         max_epoch = 10,
                         logger = logger,
                         callbacks = [early_stop_callback])

    trainer.fit(model, datamodule = dm)