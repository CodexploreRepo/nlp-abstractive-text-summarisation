#@title `NewsSummary` Model

import pytorch_lightning as pl
from transformers import BartForConditionalGeneration, AdamW
from typing import Union
from pathlib import Path

class NewsSummaryModel(pl.LightningModule):
  MODEL_BASE = BartForConditionalGeneration
  MODEL_NAME = 'facebook/bart-large'
  OPTIM = AdamW
  
  def __init__(self,
               lr:int= 0.0001,
            ):
    super(NewsSummaryModel, self).__init__()

    self.model = self.MODEL_BASE.from_pretrained(self.MODEL_NAME, return_dict=True)
    self.lr = lr

  def forward(self, input_ids, attention_mask, decoder_attention_mask, labels = None):
    output = self.model(
        input_ids,
        attention_mask = attention_mask,
        labels = labels,
        decoder_attention_mask=decoder_attention_mask
    )
    return output.loss, output.logits

  def training_step(self, batch, batch_idx):
    input_ids = batch["text_input_ids"]
    attention_mask = batch["text_attention_mask"]
    labels = batch["labels"]
    labels_attention_mask = batch["labels_attention_mask"]

    loss, outputs = self(input_ids=input_ids, 
                         attention_mask=attention_mask,
                         decoder_attention_mask=labels_attention_mask,
                         labels=labels
                         )
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    input_ids = batch["text_input_ids"]
    attention_mask = batch["text_attention_mask"]
    labels = batch["labels"]
    labels_attention_mask = batch["labels_attention_mask"]

    loss, outputs = self(input_ids=input_ids, 
                         attention_mask=attention_mask,
                         decoder_attention_mask=labels_attention_mask,
                         labels=labels
                         )
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss
  
  def configure_optimizers(self):
      return self.OPTIM(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    model = NewsSummaryModel()