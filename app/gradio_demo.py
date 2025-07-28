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
    iface = gr.Interface(
        fn=translate,
        inputs=[gr.Textbox(lines=3, label="Input"), gr.Radio(["en-ttj", "ttj-en"], value="en-ttj", label="Direction")],
        outputs=gr.Textbox(label="Translation"),
        title="Rutooro-English Translator",
    )
    iface.launch()


if __name__ == "__main__":
    main()
