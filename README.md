# Introduction
Neural transducers provide an alignment-free framework for joint automatic speech recognition (ASR) and speech translation (ST). Hierarchical transducer architectures further improve multilingual speech-to-text modeling by stacking a translation-focused encoder on top of an ASR encoder to better handle reordering. However, scaling hierarchical transducers to multilingual many-to-many settings remains challenging: fully shared models often suffer from negative transfer and unstable target-language generation, while training separate models per direction is computationally prohibitive. We propose LCMA-SRT (Language-Conditional Mixture-of-Experts Adapters for Speech Recognition and Translation), which augments a hierarchical transducer with language-conditional Mixture-of-Experts (MoE) adapters. A source-conditioned MoE adapter (SC-MoE) routes using the source-language embedding to improve acoustic–phonetic modeling and reduce cross-language interference for ASR. A target-conditioned MoE adapter (TC-MoE) routes using the desired target language to guide reordering and lexical selection and to mitigate cross-target interference in many-to-many ST. Experiments on Europarl-ST (9 languages, 72 directions) show that LCMA-SRT improves both ASR and ST within a single unified model, reducing average WER and increasing BLEU and COMET over strong hierarchical transducer baselines.
<img src="./train/LCMA-SRT.png" alt="LCMR-SRT" width="60%">
# Installation
Please refer to [document](https://k2-fsa.github.io/icefall/installation/index.html) for installation.
# Europarl-ST
Please refer to this page to download the data: [Europarl-ST](https://www.mllp.upv.es/europarl-st/) 
# Main Results
## Multilingual ASR Pretraining
**WER (%) ↓**
| Model     |   de |   en |   es |   fr |   it |   nl |   pl |   pt |   ro |  Avg |
|:----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| CR-CTC    | 24.57| 18.59| 20.76| 19.24| 17.33| 36.75| 25.28| 19.82| 18.77| 22.35|
| + MoE     | 24.39| 18.41| 20.16| 18.61| 17.28| 36.83| 24.36| 19.70| 18.79| 22.06|
| + S-Bias  | 23.89| 17.60| 19.58| 17.41| 16.73| **34.72**| 23.63| 18.21| 17.97| 21.08|
| + SC-MoE  | **23.34**| **17.45**| **19.41**| **17.34**| **16.27**| 35.20| **23.28**| **18.16**| **17.48**| **20.88**|
## Joint ASR+ST
### HENT-SRT(9次Many-to-One)
**Bleu/Comet ↑**
| src\tgt | DE | EN | ES | FR | IT | NL | PL | PT | RO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DE | \- | 17.5/0.615 | 13.3/0.531 | 12.1/0.479 | 8.7/0.504 | 16.2/0.549 | 5.9/0.521 | 12.4/0.544 | 8.3/0.545 |
| EN | 15.4/0.571 | \- | 26.0/0.641 | 24.6/0.606 | 19.0/0.625 | 21.9/0.620 | 9.7/0.584 | 23.1/0.668 | 19.8/0.680 |
| ES | 9.9/0.488 | 22.1/0.652 | \- | 20.2/0.548 | 15.7/0.571 | 15.1/0.534 | 6.9/0.546 | 22.4/0.636 | 12.2/0.589 |
| FR | 11.0/0.499 | 23.5/0.685 | 20.3/0.603 | \- | 17.6/0.603 | 16.9/0.551 | 7.4/0.555 | 23.3/0.650 | 13.0/0.618 |
| IT | 11.3/0.507 | 23.0/0.679 | 21.3/0.614 | 20.3/0.569 | \- | 16.1/0.551 | 8.3/0.568 | 22.4/0.650 | 13.4/0.623 |
| NL | 7.1/0.444 | 15.6/0.581 | 11.3/0.500 | 10.4/0.460 | 7.3/0.467 | \- | 3.7/0.486 | 10.4/0.509 | 6.3/0.509 |
| PL | 9.5/0.515 | 19.3/0.643 | 17.1/0.568 | 15.7/0.518 | 11.9/0.545 | 14.3/0.543 | \- | 14.6/0.584 | 10.0/0.583 |
| PT | 10.9/0.522 | 23.7/0.692 | 22.1/0.631 | 21.3/0.584 | 17.3/0.605 | 15.6/0.556 | 7.5/0.576 | \- | 13.9/0.636 |
| RO | 10.9/0.514 | 25.3/0.697 | 21.4/0.606 | 21.4/0.575 | 15.8/0.596 | 16.0/0.563 | 7.9/0.569 | 18.8/0.627 | \- |
| Mean | 10.7/0.507 | 21.2/0.656 | 19.1/0.587 | 18.2/0.542 | 14.2/0.565 | 16.5/0.558 | 7.2/0.550 | 18.4/0.609 | 12.1/0.598 |
**LMR (%) ↓**
| 源\目标 | DE | EN | ES | FR | IT | NL | PL | PT | RO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DE | \- | 0.08% | 0.70% | 0.64% | 0.66% | 1.00% | 0.00% | 3.39% | 1.70% |
| EN | 0.00% | \- | 0.79% | 0.25% | 0.35% | 0.65% | 0.16% | 1.98% | 1.64% |
| ES | 0.18% | 0.22% | \- | 0.18% | 0.46% | 1.01% | 0.09% | 1.19% | 0.88% |
| FR | 0.00% | 0.11% | 0.55% | \- | 0.00% | 0.26% | 0.18% | 1.55% | 1.16% |
| IT | 0.11% | 0.00% | 0.57% | 0.23% | \- | 0.36% | 0.25% | 2.20% | 0.54% |
| NL | 0.09% | 0.34% | 0.49% | 0.49% | 0.90% | \- | 0.21% | 3.18% | 1.60% |
| PL | 0.00% | 0.22% | 0.72% | 0.16% | 0.42% | 1.80% | \- | 2.08% | 1.72% |
| PT | 0.00% | 0.17% | 0.16% | 0.16% | 0.08% | 0.41% | 0.08% | \- | 0.81% |
| RO | 0.00% | 0.15% | 0.42% | 0.26% | 0.77% | 0.91% | 0.00% | 1.58% | \- |
**WER (%) ↓**
| src\tgt | DE | EN | ES | FR | IT | NL | PL | PT | RO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DE | \- | 21.80 | 26.64 | 27.12 | 27.44 | 26.43 | 26.51 | 26.51 | 26.62 |
| EN | 16.42 | \- | 17.32 | 17.56 | 17.08 | 17.45 | 17.29 | 17.18 | 17.21 |
| ES | 21.29 | 17.77 | \- | 22.25 | 22.83 | 22.68 | 21.93 | 22.82 | 22.66 |
| FR | 19.37 | 16.07 | 19.82 | \- | 20.41 | 19.30 | 19.45 | 19.86 | 20.80 |
| IT | 18.18 | 15.05 | 19.19 | 19.32 | \- | 19.06 | 18.60 | 19.00 | 19.91 |
| NL | 38.99 | 32.95 | 38.85 | 38.85 | 39.52 | \- | 38.99 | 39.32 | 39.26 |
| PL | 25.89 | 22.01 | 26.33 | 27.19 | 25.99 | 26.47 | \- | 27.13 | 27.36 |
| PT | 19.90 | 16.27 | 21.74 | 20.82 | 20.77 | 20.99 | 20.48 | \- | 20.53 |
| RO | 22.32 | 15.85 | 21.87 | 22.04 | 23.97 | 22.88 | 23.63 | 22.82 | \- |
