import pytorch_ligthnig as pl
from pytorch_ligthning.loggers import TensorBoardLogger

from utils.preprocessor_toxic import PreprocessorToxic

if __name__ == '__main__':

    dm = PreprocessorToxic()
    model = TrainerMultilabel()