# Introduction
Neural transducers provide an alignment-free framework for joint automatic speech recognition (ASR) and speech translation (ST). Hierarchical transducer architectures further improve multilingual speech-to-text modeling by stacking a translation-focused encoder on top of an ASR encoder to better handle reordering. However, scaling hierarchical transducers to multilingual many-to-many settings remains challenging: fully shared models often suffer from negative transfer and unstable target-language generation, while training separate models per direction is computationally prohibitive. We propose LCMA-SRT (Language-Conditional Mixture-of-Experts Adapters for Speech Recognition and Translation), which augments a hierarchical transducer with language-conditional Mixture-of-Experts (MoE) adapters. A source-conditioned MoE adapter (SC-MoE) routes using the source-language embedding to improve acoustic–phonetic modeling and reduce cross-language interference for ASR. A target-conditioned MoE adapter (TC-MoE) routes using the desired target language to guide reordering and lexical selection and to mitigate cross-target interference in many-to-many ST. Experiments on Europarl-ST (9 languages, 72 directions) show that LCMA-SRT improves both ASR and ST within a single unified model, reducing average WER and increasing BLEU and COMET over strong hierarchical transducer baselines.
<img src="./train/LCMA-SRT.png" alt="LCMR-SRT" width="60%">
# Installation
Please refer to [document](https://k2-fsa.github.io/icefall/installation/index.html) for installation.
# Europarl-ST
Please refer to this page to download the data: [Europarl-ST](https://www.mllp.upv.es/europarl-st/) 
# Main Results
## Multilingual ASR Pretraining:
**WER (%) ↓**
| Model     |   de |   en |   es |   fr |   it |   nl |   pl |   pt |   ro |  Avg |
|:----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| CR-CTC    | 24.57| 18.59| 20.76| 19.24| 17.33| 36.75| 25.28| 19.82| 18.77| 22.35|
| + MoE     | 24.39| 18.41| 20.16| 18.61| 17.28| 36.83| 24.36| 19.70| 18.79| 22.06|
| + S-Bias  | 23.89| 17.60| 19.58| 17.41| 16.73| **34.72**| 23.63| 18.21| 17.97| 21.08|
| + SC-MoE  | **23.34**| **17.45**| **19.41**| **17.34**| **16.27**| 35.20| **23.28**| **18.16**| **17.48**| **20.88**|

