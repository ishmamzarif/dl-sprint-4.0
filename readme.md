# team labyrinth: robust bangla long-form asr and speaker diarization

[cite_start]this repository contains the preprocessing, training, and inference pipelines developed by team labyrinth for the buet dl sprint 4.0 [cite: 1, 42, 324][cite_start]. our framework addresses the limitations of standard transformer architectures when processing bengali audio files exceeding 30-60 seconds[cite: 323, 332].

## overview
* [cite_start]long-form asr: fine-tuned whisper-medium model achieving a 0.227 wer[cite: 9, 392, 404].
* [cite_start]speaker diarization: three-stage curriculum learning using pyannote segmentation 3.0[cite: 23, 236, 406].
* [cite_start]core technology: ctc-forced word alignment for temporal accuracy and source separation (demucs) for noise robustness[cite: 103, 263, 325].

---

## methodology

### 1. asr preprocessing and training
[cite_start]standard asr models often suffer from hallucinations or loss of alignment in long-form audio[cite: 80, 337, 358]. we implemented a ctc-based pipeline to mitigate this:
* [cite_start]ctc forced alignment: we utilized the mms-300m aligner to extract precise start/end timestamps for every word in the training set[cite: 105, 375, 380].
* [cite_start]segmentation: audio was intelligently chunked into segments of strictly under 30 seconds, ensuring cuts only occur at word boundaries[cite: 139, 382, 391].
* [cite_start]fine-tuning: the bengaliai/tugstugi whisper-medium model was fine-tuned on this aligned dataset[cite: 5, 392].
* [cite_start]post-processing: the pipeline includes n-gram phrase deduplication and repeated word removal to handle model hallucinations[cite: 171, 174].

### 2. speaker diarization
[cite_start]to handle complex multi-speaker environments, we used a curriculum learning approach[cite: 235, 406]:
* [cite_start]phase 1: base adaptation: adapting the baseline model to bengali phonetic cadence using raw, noisy audio[cite: 251, 408, 410].
* [cite_start]phase 2: clean refinement: fine-tuning on audio processed by demucs to isolate distinct speaker embeddings from background noise[cite: 263, 428, 431].
* [cite_start]phase 3: dynamic augmentation: applying gain adjustments (±6.0 db) via torch_audiomentations to improve real-world robustness[cite: 271, 435, 436].

---

## performance benchmarks

### asr results (wer ↓)
| model architecture | configuration | public wer | private wer |
| :--- | :--- | :--- | :--- |
| tugstugi (proposed) | fine-tuned | 0.21988 | 0.23585 |
| tugstugi | zero-shot | 0.36142 | 0.37871 |
| bangla-asr | fine-tuned | 0.50047 | 0.54329 |
| whisper large turbo v3 | zero-shot | 0.86594 | 0.88630 |
[cite_start][cite: 145, 185, 450]

### diarization results (der ↓)
| training strategy | public der | private der |
| :--- | :--- | :--- |
| fine-tuning + data augmentation | 0.21460 | 0.32663 |
| fine-tuning + demucs refinement | 0.21621 | 0.33454 |
| normal fine-tuning (base) | 0.23147 | 0.31129 |
[cite_start][cite: 227, 229, 453]

---

## repository structure

### notebooks
* [cite_start]labyrinth_dataset_preprocessing.ipynb: scripts for ctc-forced alignment and dataset chunking[cite: 16].
* [cite_start]labyrinth_training_finetuned_whispermedium.ipynb: asr model training and evaluation[cite: 19].
* [cite_start]labyrinth_training_normaltraining: initial diarization adaptation on raw audio[cite: 33].
* [cite_start]labyrinth_training_demucs2: diarization refinement using vocal isolation[cite: 34].
* [cite_start]labyrinth_training_demucs_augmentation: final diarization robustness training[cite: 35].

### models and datasets
* [cite_start]asr model: [whisper-medium-bangla](https://huggingface.co/zarifmahir21/whisper-medium-bangla)[cite: 13].
* [cite_start]diarization model: [bengali-diarization-aug-v1](https://huggingface.co/ishmamzarif/bengali-diarization-aug-v1)[cite: 29].
* [cite_start]processed dataset: [bengali-asr-chunked](https://huggingface.co/datasets/zarifmahir21/bengali-asr-chunked)[cite: 17, 398].

---

## team members
* [cite_start]zarif ishmam [cite: 39, 312]
* [cite_start]zarif mahir [cite: 40, 308]
* [cite_start]md. ishtiak moin [cite: 39, 316]
* [cite_start]shafnan wasif [cite: 40, 318]

[cite_start]affiliation: bangladesh university of engineering and technology (buet)[cite: 41, 309].
