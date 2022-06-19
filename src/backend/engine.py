from backend.models import BartLargeSummaryModel, BartBaseSummaryModel, \
    T5BaseSummaryModel, T5SmallSummaryModel, T5V11SummaryModel
from backend.inference import Inference
from config import MODEL_CHKPT_PATH
from transformers import BartTokenizerFast, T5TokenizerFast

class BartBaseEngine:
    def __init__(self, chkpt_name):
        self.model = BartBaseSummaryModel.load_from_checkpoint(MODEL_CHKPT_PATH / f'{chkpt_name}.ckpt')   
        self.tokenizer = BartTokenizerFast.from_pretrained(BartBaseSummaryModel.MODEL_NAME)
        self.model_infer = Inference(self.model, self.tokenizer)
    
    def summarize(self, 
                  text:str, 
                  max_length: int = 150, 
                  min_length: int = 50,
                  num_beams: int = 2):
    
        return self.model_infer.summarize(text, max_length, min_length, num_beams)

class BartLargeEngine:
    def __init__(self, chkpt_name):
        self.model = BartLargeSummaryModel.load_from_checkpoint(MODEL_CHKPT_PATH / f'{chkpt_name}.ckpt')   
        self.tokenizer = BartTokenizerFast.from_pretrained(BartLargeSummaryModel.MODEL_NAME)
        self.model_infer = Inference(self.model, self.tokenizer)
    
    def summarize(self, 
                  text:str, 
                  max_length: int = 150, 
                  min_length: int = 50,
                  num_beams: int = 2):
    
        return self.model_infer.summarize(text, max_length, min_length, num_beams)

class T5BaseEngine:
    def __init__(self, chkpt_name):
        self.model = T5BaseSummaryModel.load_from_checkpoint(MODEL_CHKPT_PATH / f'{chkpt_name}.ckpt')   
        self.tokenizer = T5TokenizerFast.from_pretrained(T5BaseSummaryModel.MODEL_NAME)
        self.model_infer = Inference(self.model, self.tokenizer)
    
    def summarize(self, 
                  text:str, 
                  max_length: int = 150, 
                  min_length: int = 50,
                  num_beams: int = 2):
    
        return self.model_infer.summarize(text, max_length, min_length, num_beams)

class T5SmallEngine:
    def __init__(self, chkpt_name):
        self.model = T5SmallSummaryModel.load_from_checkpoint(MODEL_CHKPT_PATH / f'{chkpt_name}.ckpt')
        self.tokenizer = T5TokenizerFast.from_pretrained(T5SmallSummaryModel.MODEL_NAME)
        self.model_infer = Inference(self.model, self.tokenizer)
    
    def summarize(self, 
                  text:str, 
                  max_length: int = 150, 
                  min_length: int = 50,
                  num_beams: int = 2):
    
        return self.model_infer.summarize(text, max_length, min_length, num_beams)

class T5V11Engine:
    def __init__(self, chkpt_name):
        self.model = T5V11SummaryModel.load_from_checkpoint(MODEL_CHKPT_PATH / f'{chkpt_name}.ckpt')
        self.tokenizer = T5TokenizerFast.from_pretrained(T5BaseSummaryModel.MODEL_NAME)
        self.model_infer = Inference(self.model, self.tokenizer)
    
    def summarize(self, 
                  text:str, 
                  max_length: int = 150, 
                  min_length: int = 50,
                  num_beams: int = 2):
    
        return self.model_infer.summarize(text, max_length, min_length, num_beams)

if __name__ == "__main__":
    t5_base_engine = T5BaseEngine("Best-T5-epoch=0-step=10500-val_loss=1.67")
    
    text ="The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
    candidate = t5_base_engine.summarize(text, max_length=100, min_length=10, num_beams=6)
    print(candidate)