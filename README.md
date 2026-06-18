# CS596 Research Topics — Assignment 3: Multi-label Image Classification (PASCAL VOC 2007)

Coursework for **CS596 – Special Topics on Deep Learning**, Bishop's University.

Predicting which of the 20 PASCAL VOC classes are present in each image (a 20-way present/absent
multi-label task), evaluated by mean Average Precision (mAP). The solution is in
[`A3_VOC_Multilabel_Classification.ipynb`](A3_VOC_Multilabel_Classification.ipynb).

## Experiments and results (RTX 3080, 20 epochs each)

**Part A — pre-defined models** and **Part B — a self-designed model** (`MyNet`, no pre-training):

| Model | Test mAP |
|-------|---------:|
| AlexNet (from scratch) | 0.196 |
| AlexNet (ImageNet-pretrained, fine-tuned) | **0.717** |
| SimpleNet (from scratch) | 0.308 |
| MyNet — self-designed residual net (Part B) | 0.434 |

Fine-tuning the pretrained AlexNet wins by a wide margin: VOC2007 has only ~5,000 training images,
too few to learn good features from scratch. The self-designed `MyNet` (batch norm + residual
connections) trains stably from scratch and beats both from-scratch baselines, as Part B requires.

## Code

- `classifiers.py` — `SimpleNet` (Part A warm-up) and `MyNet` (Part B, self-designed residual net).
- `voc_data.py` — builds 20-dim multi-label targets and 224×224 image tensors from the VOC
  annotations (downloaded automatically by `torchvision`).

## Running it

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn   # CUDA build of torch for GPU
jupyter notebook A3_VOC_Multilabel_Classification.ipynb
```

## Files

| File | Description |
|------|-------------|
| `A3_VOC_Multilabel_Classification.ipynb` | Full solution (Parts A and B) |
| `classifiers.py` | SimpleNet and MyNet |
| `voc_data.py` | VOC multi-label data preparation |
| `Assignment 3.pdf` | Assignment description |
