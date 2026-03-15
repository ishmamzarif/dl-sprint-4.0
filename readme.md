# team labyrinth: robust bangla long-form asr and speaker diarization

this repository contains the preprocessing, training, and inference pipelines developed by team labyrinth for the buet dl sprint 4.0. our framework addresses the limitations of standard transformer architectures when processing bengali audio files exceeding 30-60 seconds.

## for details: <br>
- [slides](https://github.com/ishmamzarif/dl-sprint-4.0/blob/main/docs/labyrinth_slides.pdf) <br>
- [arxiv_paper](https://arxiv.org/abs/2602.22935)

## overview
* long-form asr: fine-tuned whisper-medium model achieving a 0.227 wer.
* speaker diarization: three-stage curriculum learning using pyannote segmentation 3.0.
* core technology: ctc-forced word alignment for temporal accuracy and source separation (demucs) for noise robustness.

---

## methodology

### 1. asr preprocessing, training, and inference
standard asr models often suffer from hallucinations or loss of alignment in long-form audio. we implemented a ctc-based pipeline to mitigate this:
* ctc forced alignment: we utilized the mms-300m aligner to extract precise start/end timestamps for every word in the training set. [see preprocessing notebook](https://github.com/ishmamzarif/dl-sprint-4.0/blob/main/labyrinth_dataset_preprocessing.ipynb)
* segmentation: audio was intelligently chunked into segments of strictly under 30 seconds, ensuring cuts only occur at word boundaries.
* fine-tuning: the bengaliai/tugstugi whisper-medium model was fine-tuned on this aligned dataset. [see asr training notebook](https://github.com/ishmamzarif/dl-sprint-4.0/blob/main/labyrinth_training_finetuned_whispermedium.ipynb)
* post-processing: the pipeline includes n-gram phrase deduplication and repeated word removal to handle model hallucinations.
* inference: batched asr inference utilizing silero vad-aware chunking to process test audio. [see asr inference notebook](https://github.com/ishmamzarif/dl-sprint-4.0/blob/main/labyrinth_asr_inference.ipynb)

### 2. speaker diarization
to handle complex multi-speaker environments, we used a curriculum learning approach:
* phase 1: base adaptation: adapting the baseline model to bengali phonetic cadence using raw, noisy audio. [see phase 1 notebook](https://github.com/ishmamzarif/dl-sprint-4.0/blob/main/labyrinth-training-normaltraining-1.ipynb)
* phase 2: clean refinement: fine-tuning on audio processed by demucs to isolate distinct speaker embeddings from background noise. [see phase 2 notebook](https://github.com/ishmamzarif/dl-sprint-4.0/blob/main/labyrinth-training-demucs2.ipynb)
* phase 3: dynamic augmentation: applying gain adjustments (±6.0 db) via torch_audiomentations to improve real-world robustness. [see phase 3 notebook](https://github.com/ishmamzarif/dl-sprint-4.0/blob/main/labyrinth-training-demucs-augmentation.ipynb)
* inference: executed on parallel gpus applying the fine-tuned pyannote model on unannotated test files. [see diarization inference notebook](https://github.com/ishmamzarif/dl-sprint-4.0/blob/main/labyrinth_diarization_inference.ipynb)

---

## performance benchmarks

### asr results (wer ↓)
| model architecture | configuration | public wer | private wer |
| :--- | :--- | :--- | :--- |
| tugstugi (proposed) | fine-tuned | 0.21988 | 0.23585 |
| tugstugi | zero-shot | 0.36142 | 0.37871 |
| bangla-asr | fine-tuned | 0.50047 | 0.54329 |
| whisper large turbo v3 | zero-shot | 0.86594 | 0.88630 |

### diarization results (der ↓)
| training strategy | public der | private der |
| :--- | :--- | :--- |
| fine-tuning + data augmentation | 0.21460 | 0.32663 |
| fine-tuning + demucs refinement | 0.21621 | 0.33454 |
| normal fine-tuning (base) | 0.23147 | 0.31129 |

---

## models and datasets
* asr model: [whisper-medium-bangla](https://huggingface.co/zarifmahir21/whisper-medium-bangla)
* diarization model: [bengali-diarization-aug-v1](https://huggingface.co/ishmamzarif/bengali-diarization-aug-v1)
* processed dataset: [bengali-asr-chunked](https://huggingface.co/datasets/zarifmahir21/bengali-asr-chunked)

---

## team members
* zarif ishmam
* zarif mahir
* md. ishtiak moin
* shafnan wasif
