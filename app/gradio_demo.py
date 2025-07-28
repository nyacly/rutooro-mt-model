"""Gradio app for Rutooro-English translation"""
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = "./model"  # path to fine-tuned model


def load_model(direction):
    if direction == 'en-ttj':
        src, tgt = 'eng_Latn', 'ttj_Latn'
    else:
        src, tgt = 'ttj_Latn', 'eng_Latn'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    tokenizer.src_lang = src
    tokenizer.tgt_lang = tgt
    return tokenizer, model


def translate(text, direction):
    tokenizer, model = load_model(direction)
    inputs = tokenizer(text, return_tensors="pt")
    output_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang])
    return tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]


def main():
    """Launch the interactive translation demo."""

    examples = [
        ["How are you?", "en-ttj"],
        ["Oraire ota?", "ttj-en"],
    ]

    iface = gr.Interface(
        fn=translate,
        inputs=[
            gr.Textbox(lines=3, label="Input Text"),
            gr.Radio(["en-ttj", "ttj-en"], value="en-ttj", label="Direction (English→Rutooro or vice versa)")
        ],
        outputs=gr.Textbox(label="Translated Text"),
        title="Rutooro-English Translator",
        examples=examples,
    )
    iface.launch()


if __name__ == "__main__":
    main()
