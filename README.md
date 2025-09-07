# Alzheimer Mixture of Experts (MoE) with Channel & Spatial Attention

This repository provides a PyTorch implementation of an **Alzheimer’s Disease classification model** based on a **Mixture of Experts (MoE)** architecture with **Channel and Spatial Attention**.
The model integrates DenseNet-inspired feature extraction, a gating mechanism for expert selection, and attention modules to improve interpretability and performance on medical imaging tasks.

---

## 🚀 Features

* **Dense Blocks** – Efficient feature reuse with dense connections.
* **Channel Attention (CA)** – Enhances informative feature channels.
* **Spatial Attention (SA)** – Focuses on key spatial regions in MRI/CT scans.
* **Mixture of Experts (MoE)** – Multiple experts with adaptive gating network.
* **Auxiliary Classifier** – Helps stabilize training with intermediate supervision.
* **Multi-output** – Provides main and auxiliary predictions along with attention maps.

## 🧠 Model Overview

The model consists of:

1. **Initial Convolution Layer** – Standard stem for feature extraction.
2. **MoE Stages (3x)** – Each stage has:

   * Multiple **Experts** (DenseBlocks + Attention).
   * **Gating Network** to assign weights to experts.
   * **Transition Layer** for downsampling.
3. **Auxiliary Classifier** – Early classification branch.
4. **Main Classifier** – Final prediction head.

---

## ▶️ Usage

Run the model script to print the architecture and test a forward pass:

```bash
python alzheimer-moe.py
```

Example output:

```
===== Model Architecture =====
... (torchsummary output)

===== Forward Pass Test =====
Main output shape: torch.Size([1, 3])
Auxiliary output shape: torch.Size([1, 3])
```

---

## 📊 Outputs

The model returns:

* **Main Output** → Final classification logits.
* **Auxiliary Output** → Intermediate classification logits.
* **Gating Weights** (optional) → Expert assignment probabilities.
* **Attention Maps** (optional) → Channel/Spatial attention visualization.

Example forward pass:

```python
from model import AlzheimerMoEModel
import torch

model = AlzheimerMoEModel(input_channels=3, num_classes=3)
dummy_input = torch.randn(1, 3, 224, 224)

main_out, aux_out, gates, attn = model(dummy_input, return_attention=True)
```

---
