# Rutooro-English Machine Translation

This repository fine-tunes the `facebook/nllb-200-distilled-600M` model on an English↔Rutooro corpus.

## Structure

```
.
├── data/                  # dataset location
├── notebooks/             # training notebook
├── scripts/               # helper scripts
├── app/                   # gradio demo
├── models/                # config / trained model
```

## Getting the data

Install dependencies first:

```bash
pip install -r requirements.txt
```

Download the raw parallel corpus from HuggingFace or MaNy-Eng and convert it to JSON:

```bash
python scripts/download_dataset.py --source hf --output data/english_rutooro.json
```

Clean the data and create train/dev/test splits (files will be written to `data/clean/`):

```bash
python scripts/preprocess.py data/english_rutooro.json data/clean
```

## Training

Open `notebooks/train_nllb_colab.ipynb` in Google Colab. The notebook checks GPU availability, enables mixed precision training and uses early stopping. Mount your Google Drive when prompted so training data and checkpoints are stored persistently.

After mounting, the notebook creates the following folders in your Drive:

```
/content/drive/MyDrive/rutooro-mt-data/    # datasets
/content/drive/MyDrive/rutooro-mt-models/  # model checkpoints
/content/drive/MyDrive/rutooro-mt-outputs/ # logs and metrics
```

Prepared datasets, checkpoints and final models will be saved here so you can resume training or run the demo without reprocessing.

## Demo

Launch a small Gradio interface to test the trained model locally:

```bash
python app/gradio_demo.py
```

## Sample translations

| English | Rutooro |
|---------|---------|
| "How are you?" | "Oraire ota?" |
| "Thank you" | "Webale" |

