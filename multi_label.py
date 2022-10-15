import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils.preprocessor_toxic import PreprocessorToxic
from utils.trainer_multilabel import TrainerMultilabel

if __name__ == '__main__':

    dm = PreprocessorToxic()
    model = TrainerMultilabel()

    early_stop_callback = EarlyStopping(monitor = 'val_loss', 
                                        min_delta = 0.00,
                                        patience = 3,
                                        mode = "min")
    
    logger = TensorBoardLogger("logs", name="bert_multilabel")

    trainer = pl.Trainer(gpus = 1,
                         max_epochs = 10,
                         logger = logger,
                         default_root_dir = "./checkpoints/class",
                         callbacks = [early_stop_callback])

    trainer.fit(model, datamodule = dm)