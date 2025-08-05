# Arabic Text Filter Pipeline

This Python pipeline filters Arabic text based on:

1. **Length and language constraints** (Arabic only, within a word count range),
2. **Arabic dialect identification**, and
3. **Hate speech removal**.

It utilizes **LLaMA via Ollama** for dialect classification and **SetFit** for hate speech detection.

---

## Features

* ✅ Filters by Arabic word count and Arabic script only.
* ✅ Identifies Arabic dialects (`msa`, `egy`, `lev`, `glf`, `mag`, `irq`) using LLaMA.
* ✅ Filters out hate speech using the `akhooli/setfit_ar_hs` model.
* ✅ Supports `.txt`, `.csv`, `.xlsx` input files.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/arabic-text-filter.git
cd arabic-text-filter
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
pip install pandas tqdm setfit lightrag[ollama]
```

> **Note:** SetFit uses `scikit-learn`. You may get a version warning if your `scikit-learn` version differs from the model's. For compatibility, use `scikit-learn==1.2.2` if needed.

---

## Ollama Setup

### 1. Start the Ollama Server

```bash
ollama serve &
```

### 2. Download LLaMA Model

You can choose any Ollama-supported model. This example uses `llama3.1:8b`:

```bash
ollama pull llama3.1:8b
```

### 3. Verify Ollama is Running

```bash
curl http://127.0.0.1:11434
```

A response with HTTP status `200 OK` means the server is active.

---

## Usage

### Example Command

```bash
python3 filter_script.py \
  --input data.txt \
  --columns comment \
  --target_dialect egy \
  --remove_hate
```

### Command-line Arguments

| Argument           | Type   | Description                                                          |
| ------------------ | ------ | -------------------------------------------------------------------- |
| `--input`          | str    | Path to the input file (`.txt`, `.csv`, `.xlsx`).                    |
| `--output`         | str    | Output file name (default: `filtered_output.txt`).                   |
| `--min_words`      | int    | Minimum number of words per line (default: `10`).                    |
| `--max_words`      | int    | Maximum number of words per line (default: `20`).                    |
| `--columns`        | str\[] | Column names to read (for CSV/XLSX files). Multiple columns allowed. |
| `--target_dialect` | str    | Arabic dialect to retain (`msa`, `egy`, `lev`, `glf`, `mag`, `irq`). |
| `--remove_hate`    | flag   | If set, removes sentences predicted as hate speech.                  |

---

## Output

* A text file (`filtered_output.txt` by default) containing **filtered lines**.
* Printed logs indicating:

  * Number of lines kept after each filtering stage.
  * Any errors encountered during model inference.

---

## Troubleshooting

### 1. **Port Already in Use**

If you see:

```
listen tcp 0.0.0.0:11434: bind: address already in use
```

Ollama is already running. Do **not** run `ollama serve` again.

### 2. **Model Not Found**

If you see:

```
Error calling the model: model 'llama3.1:8b' not found
```

Ensure you downloaded the model:

```bash
ollama pull llama3.1:8b
```

### 3. **SetFit Sklearn Warning**

```
InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression...
```

Ignore for now, or use `scikit-learn==1.2.2` to match the model's original version.

---

## License

[MIT License](LICENSE)

---

## Credits

* [Ollama](https://ollama.com) for LLaMA model inference.
* [SetFit](https://github.com/huggingface/setfit) for zero-shot classification.
* Hate speech detection model: [akhooli/setfit\_ar\_hs](https://huggingface.co/akhooli/setfit_ar_hs)

---

Let me know if you'd like the README in Markdown (`.md`) format. I can generate it as a downloadable file for you.
