from backend.models import BartLargeSummaryModel
from backend.inference import Inference
from config import MODEL_CHKPT_PATH
from transformers import BartTokenizerFast

class BartLargeEngine:
    def __init__(self, chkpt_name):
        self.bart_large_model = BartLargeSummaryModel.load_from_checkpoint(MODEL_CHKPT_PATH / f'{chkpt_name}.ckpt')   
        self.tokenizer = BartTokenizerFast.from_pretrained(BartLargeSummaryModel.MODEL_NAME)
        self.model_infer = Inference(self.bart_large_model, self.tokenizer)
    
    def summarize(self, 
                  text:str, 
                  max_length: int = 150, 
                  num_beams: int = 2):
    
        return self.model_infer.summarize(text, max_length, num_beams)


if __name__ == "__main__":
    bart_l_engine = BartLargeEngine("BART-large-epoch=0-step=500-val_loss=0.91")
    
    text ="The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
    candidate = bart_l_engine.summarize(text, max_length=200, num_beams=4)
    print(candidate)