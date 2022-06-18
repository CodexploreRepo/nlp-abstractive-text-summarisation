from backend.concept.abstract_model import NewsSummaryModel
from transformers import (
    BartForConditionalGeneration
)
from typing import Literal
from config import MODEL_CHKPT_PATH

class BartLargeSummaryModel(NewsSummaryModel):
    MODEL_BASE = BartForConditionalGeneration
    MODEL_NAME = 'facebook/bart-large'
    def __init__(self, 
                 lr: int = 0.0001,
                ):
        super(BartLargeSummaryModel, self).__init__(
            lr = lr
        )


    

if __name__ == "__main__":
    bart_large = BartLargeSummaryModel.load_from_checkpoint(MODEL_CHKPT_PATH / 'BART-large-epoch=0-step=500-val_loss=0.91.ckpt')   
        
