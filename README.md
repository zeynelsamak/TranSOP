# TranSOP: Transformer-based Multimodal Classification for Stroke Treatment Outcome Prediction [[Paper]](https://arxiv.org/pdf/2301.10829.pdf)
by [Zeynel Abidin Samak](https://scholar.google.co.uk/citations?user=QOrEQ3AAAAAJ&hl=en), [Philip Clatworthy](https://scholar.google.co.uk/citations?user=B6lFOAQAAAAJ&hl=en) and [Majid Mirmehdi](https://scholar.google.com/citations?user=NsW3yAwAAAAJ&hl=en)


## Abstract
  Acute ischaemic stroke, caused by an interruption in blood flow to brain tissue, is a leading cause of disability and mortality worldwide. The selection of patients for the most optimal ischaemic stroke treatment is a crucial step for a successful outcome, as the effect of treatment highly depends on the time to treatment. We propose a transformer-based multimodal network (TranSOP) for a classification approach that employs clinical metadata and imaging information, acquired on hospital admission, to predict the functional outcome of stroke treatment based on the modified Rankin Scale (mRS). This includes a fusion module to efficiently combine 3D non-contrast computed tomography (NCCT) features and clinical information. In comparative experiments using unimodal and multimodal data on the MRCLEAN dataset, we achieve a state-of-the-art AUC score of 0.85.

## Code
Feel free to contact if you may have any questions about the code.

### Hyperparameters
| Parameter                | Value            |
|--------------------------|------------------|
| Number of epochs         | 500              |
| Batch size               | 24               |
| Learning rate 0.0003     | 0.0003           |
| Learning rate scheduler  | Cosine annealing |
| Optimizer                | Adam             |
| Weight decay             | 0.0001           |
| Loss Function            | Focal Loss       |


## Cite

```latex
@inproceedings{samak2023transop,
  title={TranSOP: Transformer-based Multimodal Classification for Stroke Treatment Outcome Prediction},
  author={Samak, Zeynel A and Clatworthy, Philip L and Mirmehdi, Majid},
  booktitle={20th IEEE International Symposium on Biomedical Imaging, ISBI 2023},
  year={2023},
  organization={IEEE Computer Society}
}
```

