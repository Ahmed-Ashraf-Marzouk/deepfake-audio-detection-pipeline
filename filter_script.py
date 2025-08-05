import re
import os
import argparse
import subprocess
import threading
import socket
import pandas as pd
from tqdm import tqdm
from unicodedata import normalize
from lightrag.core.generator import Generator
from lightrag.components.model_client import OllamaClient
from setfit import SetFitModel


class ArabicTextFilterPipeline:
    ARABIC_UNICODE_PATTERN = re.compile(r'^[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s]+$')
    DIALECTS = ['msa', 'egy', 'lev', 'glf', 'mag', 'irq']

    def __init__(self, input_file, output_file="filtered_output.txt", min_words=10, max_words=20,
                 columns=None, target_dialect=None, remove_hate=False):
        self.input_file = input_file
        self.output_file = output_file
        self.min_words = min_words
        self.max_words = max_words
        self.columns = columns
        self.target_dialect = target_dialect
        self.remove_hate = remove_hate
        self.lines = []
        self.filtered_lines = []

        self._start_ollama_server_if_needed()
        self.llama_generator = self._init_llama_generator()
        self.setfit_model = SetFitModel.from_pretrained("akhooli/setfit_ar_hs")

    def _start_ollama_server_if_needed(self):
        def is_port_open(host, port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.connect((host, port))
                    return True
                except (ConnectionRefusedError, OSError):
                    return False

        if not is_port_open('localhost', 11434):
            print("Starting Ollama server...")
            def ollama():
                os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
                os.environ['OLLAMA_ORIGINS'] = '*'
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            threading.Thread(target=ollama, daemon=True).start()

    def _init_llama_generator(self):
        prompt_template = """<SYS>
You are a linguistics expert. Classify the Arabic dialect in the sentence.
Valid dialects are: msa, egy, lev, glf, mag, irq.
</SYS>
User: Sentence: {{input_str}}
What Arabic dialect is this written in?
You:"""
        return Generator(
            model_client=OllamaClient(),
            model_kwargs={"model": "llama3.1:8b"},
            template=prompt_template
        )

    def load_data(self):
        ext = os.path.splitext(self.input_file)[1].lower()
        print(f"Loading data from {self.input_file}...")
        if ext == '.txt':
            with open(self.input_file, 'r', encoding='utf-8') as f:
                self.lines = [line.strip() for line in f if line.strip()]
        elif ext in ['.csv', '.xls', '.xlsx']:
            df = pd.read_csv(self.input_file) if ext == '.csv' else pd.read_excel(self.input_file)
            if self.columns:
                missing = [col for col in self.columns if col not in df.columns]
                if missing:
                    raise ValueError(f"Column(s) not found: {missing}")
                selected = df[self.columns]
            else:
                selected = df.iloc[:, [0]]
            self.lines = selected.fillna('').astype(str).agg(' '.join, axis=1).tolist()
        else:
            raise ValueError("Unsupported file format. Supported: .txt, .csv, .xlsx")
        print(f"✓ Loaded {len(self.lines)} lines.")

    def filter_text(self):
        print("Filtering Arabic text and by word count...")
        for line in tqdm(self.lines, desc="Arabic Filtering"):
            text = line.strip()
            word_count = len(text.split())
            if not (self.min_words <= word_count <= self.max_words):
                continue
            if not self.ARABIC_UNICODE_PATTERN.fullmatch(text):
                continue
            self.filtered_lines.append(text)
        print(f"✓ Kept {len(self.filtered_lines)} lines after filtering.")

    def filter_by_dialect(self):
        if not self.target_dialect:
            return
        print(f"Filtering by dialect: {self.target_dialect}")
        result_lines = []
        for text in tqdm(self.filtered_lines, desc="Dialect Filtering"):
            try:
                response = self.llama_generator.call({"input_str": text})
                pred = response.data.strip().lower()
                dialect_match = next((d for d in self.DIALECTS if d in pred), 'unknown')
                if dialect_match == self.target_dialect:
                    result_lines.append(text)
            except Exception as e:
                print(f"[Warning] Skipping text due to error: {e}")
        self.filtered_lines = result_lines
        print(f"✓ Kept {len(self.filtered_lines)} lines after dialect filtering.")

    def filter_hate_speech(self):
        if not self.remove_hate or not self.filtered_lines:
            print("Skipping hate speech filtering (no lines or disabled).")
            return
        print("Filtering hate speech...")
        normalized_texts = [normalize('NFKC', text) for text in self.filtered_lines]
        preds = self.setfit_model.predict(normalized_texts)
        result_lines = [text for text, label in zip(self.filtered_lines, preds) if label != 'hate_speech']
        self.filtered_lines = result_lines
        print(f"✓ Kept {len(self.filtered_lines)} lines after hate speech filtering.")

    def save_output(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.writelines(line + '\n' for line in self.filtered_lines)
        print(f"💾 Output saved to {self.output_file}")

    def run_pipeline(self):
        self.load_data()
        self.filter_text()
        self.filter_by_dialect()
        self.filter_hate_speech()
        self.save_output()


# === CLI ===
def main():
    parser = argparse.ArgumentParser(description="Arabic Text Filter Pipeline (LLaMA + SetFit)")
    parser.add_argument("--input", type=str, required=True, help="Input file (.txt, .csv, .xlsx)")
    parser.add_argument("--output", type=str, default="filtered_output.txt", help="Output text file")
    parser.add_argument("--min_words", type=int, default=10, help="Minimum words per line")
    parser.add_argument("--max_words", type=int, default=20, help="Maximum words per line")
    parser.add_argument("--columns", type=str, nargs='*', default=None, help="Column names to read (for CSV/XLSX)")
    parser.add_argument("--target_dialect", type=str, choices=ArabicTextFilterPipeline.DIALECTS,
                        help="Keep only this dialect (msa, egy, lev, glf, mag, irq)")
    parser.add_argument("--remove_hate", action='store_true', help="Remove hate speech sentences")

    args = parser.parse_args()

    pipeline = ArabicTextFilterPipeline(
        input_file=args.input,
        output_file=args.output,
        min_words=args.min_words,
        max_words=args.max_words,
        columns=args.columns,
        target_dialect=args.target_dialect,
        remove_hate=args.remove_hate
    )
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
