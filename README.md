# Rutooro-English Machine Translation

This repository fine-tunes the `facebook/nllb-200-distilled-600M` model on a small English↔Rutooro corpus (~16k sentence pairs).

## Structure

```
.
├── data/                  # dataset location
├── notebooks/             # training notebook
├── scripts/               # helper scripts
├── app/                   # gradio demo
├── models/                # config / trained model
```

## Usage

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Run preprocessing or evaluation scripts in `scripts/`.
3. Launch the Gradio demo:
   ```bash
   python app/gradio_demo.py
   ```

See `notebooks/train_nllb_colab.ipynb` for a full training pipeline on Google Colab.
