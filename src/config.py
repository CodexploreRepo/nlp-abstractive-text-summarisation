# Global Config
from pathlib import Path

PAGE_TITLE = "Text Sumarization"

src = Path(__file__).resolve().parents[0]
MODEL_CHKPT_PATH = src / "backend" / "chkpt"

if __name__ == "__main__":
    print(model_chkpt)