"""Preprocess dataset for Rutooro-English translation."""
import json
import argparse
from pathlib import Path


def preprocess(input_path: str, output_path: str):
    """Load dataset from json (list of {\"translation\":{\"en\":...,\"tt\":...}}) and save parallel pairs."""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    for row in dataset:
        if 'translation' in row:
            en = row['translation'].get('english') or row['translation'].get('en')
            tt = row['translation'].get('rutooro') or row['translation'].get('ttj') or row['translation'].get('tt')
            if en and tt:
                data.append({'translation': {'en': en, 'ttj': tt}})
    Path(output_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input JSON file')
    parser.add_argument('output', help='Output cleaned JSON file')
    args = parser.parse_args()
    preprocess(args.input, args.output)
