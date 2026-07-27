# Offical repo for "PGSF: Profile-Guided Semantic Fusion for Decoupled Industrial Pre-Ranking"

**PGSF: Profile-Guided Semantic Fusion for Decoupled Industrial Pre-Ranking (SIGIR 2026)**  
*Chuike Sun, Nan Xu, Xing Fang, Yang Huang, Jing Wang, Junzhou Chen*  
[DOI Link](https://doi.org/10.1145/3805712.3808480)  

## **Abstract**  
Pre-ranking is a critical stage in large-scale recommendation systems, strictly demanding both high efficiency and effectiveness. Despite being the industry benchmark for pre-ranking owing to its serving efficiency, the two-tower structure, by virtue of its decoupled architecture, inherently restricts fine-grained user-item semantic interactions. In this paper, we propose Profile-Guided Semantic Fusion (PGSF), a plug-and-play framework that injects target-item guidance during training while preserving tower decoupling at inference. PGSF comprises three modules: (1) Profile-Item Semantic Alignment (PISA) aligns user profile semantics with the target item via contrastive learning to generate a profile-guided query; (2) Multi-Interest Semantic Modeling (MISM) constructs a discrete semantic space using RQ-VAE to explicitly model heterogeneous user interests; (3) Multi-Level Adaptive Fusion (MLAF) hierarchically retrieves target-aware interests from historical behaviors and semantic candidates, then fuses them with the backbone user representation to produce a fine-grained enhanced user representation. Training jointly optimizes ranking, alignment, and interest prediction losses. Evaluated on Tmall App, PGSF integrates smoothly into existing decoupled pre-ranking systems, achieving +2.66% uCTR and +3.13% IPV gains. The implementation is available at https://github.com/SunChuike/PGSF.

## Release Notes
The implementation of PGSF is based on our company's customized distributed TensorFlow framework, designed to optimize industrial applications. Due to company policy, this repository provides a carefully extracted and simplified version of the source code. While it is not runnable out-of-the-box, it is intended as supporting material to clearly illustrate the implementation logic of the model architecture and key modules, thereby enhancing the transparency of our method's design.

## Model Architecture & Key Components
The PGSF architecture, highlighting critical modules, is shown below. Accompanying code snippets illustrate their implementation logic.

*   **Overall Model Structure:**

<img width="2227" height="1012" alt="model" src="https://github.com/user-attachments/assets/0a042c8c-99d8-4575-9f6c-22502e3bbf37" />

*   **Annotated Code Snippet for Key Modules (model.py):**

<img width="1532" height="596" alt="key module" src="https://github.com/user-attachments/assets/a08b99a5-d199-4d6a-b25a-59e496887745" />

## **Citation**  
If you use PGSF, please cite:  

```bibtex
@inproceedings{10.1145/3805712.3808480,
author = {Sun, Chuike and Xu, Nan and Fang, Xing and Huang, Yang and Wang, Jing and Chen, Junzhou},
title = {PGSF: Profile-Guided Semantic Fusion for Decoupled Industrial Pre-Ranking},
year = {2026},
isbn = {9798400725999},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3805712.3808480},
doi = {10.1145/3805712.3808480},
abstract = {Pre-ranking is a critical stage in large-scale recommendation systems, strictly demanding both high efficiency and effectiveness. Despite being the industry benchmark for pre-ranking owing to its serving efficiency, the two-tower structure, by virtue of its decoupled architecture, inherently restricts fine-grained user-item semantic interactions. In this paper, we propose Profile-Guided Semantic Fusion (PGSF), a plug-and-play framework that injects target-item guidance during training while preserving tower decoupling at inference. PGSF comprises three modules: (1) Profile-Item Semantic Alignment (PISA) aligns user profile semantics with the target item via contrastive learning to generate a profile-guided query; (2) Multi-Interest Semantic Modeling (MISM) constructs a discrete semantic space using RQ-VAE to explicitly model heterogeneous user interests; (3) Multi-Level Adaptive Fusion (MLAF) hierarchically retrieves target-aware interests from historical behaviors and semantic candidates, then fuses them with the backbone user representation to produce a fine-grained enhanced user representation. Training jointly optimizes ranking, alignment, and interest prediction losses. Evaluated on Tmall App, PGSF integrates smoothly into existing decoupled pre-ranking systems, achieving +2.66\% uCTR and +3.13\% IPV gains. The implementation is available at https://github.com/SunChuike/PGSF.},
booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {4907–4911},
numpages = {5},
keywords = {recommender systems, pre-ranking system, neural networks},
location = {Australia},
series = {SIGIR '26}
}
```
## **Contact**  
📧 Email: **sunchuike.sck@taobao.com**  

