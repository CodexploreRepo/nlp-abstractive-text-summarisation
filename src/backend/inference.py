# @title `NewsSummary` Model Inference { form-width: "30%" }
import torch
import datasets
import config
import pandas as pd
from backend.concept.abstract_model import NewsSummaryModel
from typing import Union, List

class Inference:
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

  def __init__(self,
               model: NewsSummaryModel,
               tokenizer,
               text_max_token_len: int = 512,
               summary_max_token_len: int = 128, 
            ):
    self.model = model
    self.model.to(self.DEVICE)   
    self.model.eval()
    self.tokenizer = tokenizer
    self.text_max_token_len = text_max_token_len
    self.summary_max_token_len = summary_max_token_len

  def encoding_plus(self, text: str):
    return self.tokenizer(
        text,
        max_length=self.text_max_token_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

  def summarize(self, 
                text:str, 
                max_length: int = 150, 
                min_length: int = 50,
                num_beams: int = 2):
    text_encoding = self.encoding_plus(text).to(self.DEVICE)
    generated_ids = self.model.model.generate(
        input_ids = text_encoding['input_ids'],
        attention_mask = text_encoding['attention_mask'],
        max_length = max_length,
        min_length = min_length,
        num_beams = num_beams,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    preds = [
        self.tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in generated_ids
    ]
    return " ".join(preds)


if __name__ == "__main__":
    from backend.models import BartLargeSummaryModel
    from config import MODEL_CHKPT_PATH
    from transformers import BartTokenizerFast
    bart_large_model = BartLargeSummaryModel.load_from_checkpoint(MODEL_CHKPT_PATH / 'BART-large-epoch=0-step=500-val_loss=0.91.ckpt')   
    tokenizer = BartTokenizerFast.from_pretrained(BartLargeSummaryModel.MODEL_NAME)
    
    model_infer = Inference(bart_large_model, tokenizer)
    text ="The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
    candidate = model_infer.summarize(text)
    print(candidate)


