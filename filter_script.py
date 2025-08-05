import re
import os
import argparse
import subprocess
import threading
import pandas as pd
from tqdm import tqdm
from unicodedata import normalize
from lightrag.core.generator import Generator
from lightrag.components.model_client import OllamaClient
from setfit import SetFitModel

class ArabicTextFilterPipeline:
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

        self.arabic_pattern = re.compile(r'^[\u0600-\u06FF\s]+$')

        # Initialize models
        # import os
        # def ollama():
        #     os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
        #     os.environ['OLLAMA_ORIGINS'] = '*'
        #     subprocess.Popen(["ollama", "serve"])

        # ollama_thread = threading.Thread(target=ollama)
        # ollama_thread.start()
        self.llama_generator = Generator(
            model_client=OllamaClient(),
            model_kwargs={"model": "llama3.1:8b"},
            template="""<SYS>
You are a linguistics expert. Classify the Arabic dialect in the sentence.
Valid dialects are: msa, egy, lev, glf, mag, irq.
</SYS>
User: Sentence: {{input_str}}
What Arabic dialect is this written in?
You:"""
        )
        self.setfit_model = SetFitModel.from_pretrained("akhooli/setfit_ar_hs")

    def load_data(self):
        ext = os.path.splitext(self.input_file)[1].lower()
        if ext == '.txt':
            with open(self.input_file, 'r', encoding='utf-8') as f:
                self.lines = [line.strip() for line in f.readlines()]
        elif ext in ['.csv', '.xls', '.xlsx']:
            if ext == '.csv':
                df = pd.read_csv(self.input_file)
            else:
                df = pd.read_excel(self.input_file)
            if self.columns:
                try:
                    selected = df[self.columns]
                except KeyError as e:
                    raise ValueError(f"Column(s) not found: {e}")
            else:
                selected = df.iloc[:, [0]]
            self.lines = selected.fillna('').astype(str).agg(' '.join, axis=1).tolist()
        else:
            raise ValueError("Unsupported file format. Supported: .txt, .csv, .xlsx")

    def filter_text(self):
        print("Filtering Arabic text...")
        self.filtered_lines = []
        for line in tqdm(self.lines, desc="Arabic Filtering"):
            text = line.strip()
            word_count = len(text.split())
            if word_count < self.min_words or word_count > self.max_words:
                continue
            if not self.arabic_pattern.fullmatch(text):
                continue
            self.filtered_lines.append(text)
        print(f"✓ Kept {len(self.filtered_lines)} lines after Arabic and word count filtering.")

    def filter_by_dialect(self):
        if not self.target_dialect:
            return
        print(f"Filtering by dialect: {self.target_dialect}")
        result_lines = []
        for text in tqdm(self.filtered_lines, desc="Dialect Filtering"):
            pred = self.llama_generator.call({"input_str": text}).data.strip().lower()
            for dialect in ['msa', 'egy', 'lev', 'glf', 'mag', 'irq']:
                if dialect in pred:
                    pred = dialect
                    break
            else:
                pred = 'unknown'
            if pred == self.target_dialect:
                result_lines.append(text)
        print(f"✓ Kept {len(result_lines)} lines after dialect filtering.")
        self.filtered_lines = result_lines

    def filter_hate_speech(self):
        if not self.remove_hate:
            return
        print("Filtering hate speech...")
        normalized = [normalize('NFKC', text) for text in self.filtered_lines]
        preds = self.setfit_model.predict(normalized)
        result_lines = [text for text, pred in zip(self.filtered_lines, preds) if pred != 'hate_speech']
        print(f"✓ Kept {len(result_lines)} lines after hate speech filtering.")
        self.filtered_lines = result_lines

    def save_output(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for line in self.filtered_lines:
                f.write(line + '\n')
        print(f"💾 Output saved to {self.output_file}")

    def run_pipeline(self):
        self.load_data()
        self.filter_text()
        self.filter_by_dialect()
        self.filter_hate_speech()
        self.save_output()

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic Text Filter Pipeline (LLaMA + SetFit)")
    parser.add_argument("--input", type=str, required=True, help="Input file (.txt, .csv, .xlsx)")
    parser.add_argument("--output", type=str, default="filtered_output.txt", help="Output text file")
    parser.add_argument("--min_words", type=int, default=10, help="Minimum words per line")
    parser.add_argument("--max_words", type=int, default=20, help="Maximum words per line")
    parser.add_argument("--columns", type=str, default=None, nargs='*', help="Column names to read (for CSV/XLSX)")
    parser.add_argument("--target_dialect", type=str, help="Keep only this dialect (e.g., 'msa', 'egy')")
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
