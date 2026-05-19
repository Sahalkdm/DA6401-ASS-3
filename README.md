# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Links
* **Personal Repository:** [GitHub Repo Link](https://github.com/Sahalkdm/DA6401-ASS-3)
* **Project Report:** [Wandb Report Link](https://api.wandb.ai/links/sahal_k-indian-institute-of-technology-madras/w3kwu5gv)

Report Link: https://api.wandb.ai/links/sahal_k-indian-institute-of-technology-madras/w3kwu5gv

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
