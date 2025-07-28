"""Command line translation demo using a fine-tuned model."""
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(model_dir: str, src_lang: str, tgt_lang: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    return tokenizer, model


def translate(text: str, tokenizer, model, max_length: int = 128):
    inputs = tokenizer(text, return_tensors="pt")
    output_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang], max_length=max_length)
    return tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser(description="Translate text")
    parser.add_argument('model_dir', help='Path to saved model')
    parser.add_argument('text', help='Text to translate')
    parser.add_argument('--src_lang', default='eng_Latn')
    parser.add_argument('--tgt_lang', default='ttj_Latn')
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir, args.src_lang, args.tgt_lang)
    result = translate(args.text, tokenizer, model)
    print(result)


if __name__ == '__main__':
    main()
