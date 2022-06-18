# Global Config
from pathlib import Path

PAGE_TITLE = "Text Sumarization"

src = Path(__file__).resolve().parents[0]
MODEL_CHKPT_PATH = src / "backend" / "chkpt"
MODEL_CHKPT_MAP = {
    "t5-base": MODEL_CHKPT_PATH / "Best-T5-epoch=0-step=10000-val_loss=1.57",
    "facebook/bart-large": MODEL_CHKPT_PATH / "Best-BART_large-epoch=0-step=500-val_loss=0.91"
}

if __name__ == "__main__":
    print(src)