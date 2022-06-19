from backend.concept.abstract_model import NewsSummaryModel
from transformers import (
    BartForConditionalGeneration,
    T5ForConditionalGeneration
)
from typing import Literal
from config import MODEL_CHKPT_PATH

class BartBaseSummaryModel (NewsSummaryModel):
    MODEL_BASE = BartForConditionalGeneration
    MODEL_NAME = 'facebook/bart-base'
    def __init__(self, 
                 lr: int = 0.0001,
                ):
        super(BartBaseSummaryModel, self).__init__(
            lr = lr
        )

class BartLargeSummaryModel(NewsSummaryModel):
    MODEL_BASE = BartForConditionalGeneration
    MODEL_NAME = 'facebook/bart-large'
    def __init__(self, 
                 lr: int = 0.0001,
                ):
        super(BartLargeSummaryModel, self).__init__(
            lr = lr
        )

class T5BaseSummaryModel(NewsSummaryModel):
    MODEL_BASE = T5ForConditionalGeneration
    MODEL_NAME = 't5-base'
    def __init__(self, 
                 lr: int = 0.0001,
                ):
        super(T5BaseSummaryModel, self).__init__(
            lr = lr
        )   

class T5SmallSummaryModel (NewsSummaryModel):
    MODEL_BASE = T5ForConditionalGeneration
    MODEL_NAME = 't5-small'
    def __init__(self, 
                 lr: int = 0.0001,
                ):
        super(T5SmallSummaryModel, self).__init__(
            lr = lr
        )

class T5V11SummaryModel (NewsSummaryModel):
    MODEL_BASE = T5ForConditionalGeneration
    MODEL_NAME = 'google/t5-v1_1-small'
    def __init__(self, 
                 lr: int = 0.0001,
                ):
        super(T5V11SummaryModel, self).__init__(
            lr = lr
        )


if __name__ == "__main__":
    #bart_large = BartLargeSummaryModel.load_from_checkpoint(MODEL_CHKPT_PATH / 'BART-large-epoch=0-step=500-val_loss=0.91.ckpt')   
    t5_base = T5BaseSummaryModel.load_from_checkpoint(MODEL_CHKPT_PATH / 'Best-T5-epoch=0-step=10500-val_loss=1.67.ckpt')   
            
