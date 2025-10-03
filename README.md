# SentOrd
A sequence-to-sequence model that restores shuffled sentences into their correct order. Built by fine-tuning **T5-small** on the **CNN/DailyMail dataset**. The project demonstrates text coherence modeling and narrative reconstruction.

---

## Overview

Sentence ordering (also called text coherence modeling) is the task of rearranging shuffled sentences into a coherent order.
This project fine-tunes **T5-small** to predict the correct ordering of sentences from scrambled input.

---

## Project Structure

```
SentOrd/
│── SentOrd.ipynb    # Main notebook for training, evaluation, and inference
│── README.md
```

---

## Setup

Clone the repo and install dependencies:

```bash
git clone <repo-url>
cd SentOrd
pip install -r requirements.txt   # if you export env from Colab/Jupyter
```

Or simply open `SentOrd.ipynb` in **Google Colab / Jupyter** and run all cells.

---

## Dataset

* **Source**: CNN/DailyMail dataset.
* **Preprocessing**: Paragraphs were split into sentences, then randomly shuffled to create inputs.
* **Training Format**:

```
Input:  "Sentence 3 <sep> Sentence 1 <sep> Sentence 2"  
Target: "Sentence 1 <sep> Sentence 2 <sep> Sentence 3"
```

---

## Training & Evaluation

All training and evaluation is included inside **`SentOrd.ipynb`**.

* Model: T5-small (fine-tuned)
* Dataset: CNN/DailyMail
* Metric: Kendall’s Tau + Exact Match

---

## Inference

You can load the fine-tuned model from Google Drive:

**Model Weights**: [Google Drive Link](https://drive.google.com/drive/folders/1s_bG2kyuky_c9Try6vVSJIFFiVdfz_O9?usp=sharing)

Example usage inside the notebook:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("<google-drive-path>")

input_text = "Sentence 3 <sep> Sentence 1 <sep> Sentence 2"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Results

* **Kendall’s Tau**: - on CNN/DailyMail validation set
* **Exact Match Accuracy**: -
* The model demonstrates fair ability to reconstruct narrative order.

---

## Future Work

* Try larger models (T5-base, BART)
* Apply to other datasets (ROCStories, arXiv abstracts)
* Build a small web demo (e.g., with Gradio)
* Increase sentence ordering accuracy
