# Inconsistency Masks

We are excited to introduce Inconsistency Masks (IM), a new SSL approach for image segmentation. A detailed paper, elaborating on IM and our comparative study, is currently in the final stages of refinement. As soon as it becomes publicly available, a link to the paper will be provided here.
Due to limited hardware resources and the scale of our study, which required training thousands of U-Nets, we could only use small datasets and train tiny U-Nets (0.17 - 2.72 million parameters) compared to modern ViTs. But the uniform training conditions across all approaches ensure that our findings are still valuable and comparable.

![main_results](main_results.png)

In this diagram, we focus on results that exceed those of Labeled Dataset Training (LDT) to avoid excessively compressing the representation of other outcomes. For a complete overview, a diagram with all results will be available in the appendix of the paper. The benchmarks used include Full Dataset Training (FDT), Labeled Dataset Training (LDT, randomly selected 10% of the complete dataset), and Augmented Labeled Dataset Training (ALDT, Labeled Dataset with 9 additional augmented versions of each image). The SSL approaches should surpass ALDT to justify their added complexity.
Among common SSL approaches – Model Ensemble, Input Ensemble, Consistency Loss, and Noisy Student – all but Noisy Student proved to be of limited effectiveness.

**EvalNet:** This approach, inspired by the ValueNet from the AlphaGo paper, to our knowledge, has not yet been used for image segmentation. EvalNet assesses the segmentation quality of pseudo-labels, using only those segmentation masks for training that exceed a set threshold in IoU/mIoU score.

**IM:** Our novel approach demonstrates strong performance, consistently outperforming all other SSL methods across various datasets, particularly in the initial generations. The only exception is the Noisy Student method, which manages to match or surpass IM after four Generations in ISIC 2018, albeit with approximately six times more parameters.

**Combination Approaches:** IM+ combines IM with Noisy Student. In AIM+, the starting point is the best model from ALDT, not LDT, and the Labeled Dataset is replaced by the Augmented Labeled Dataset. In IM++ and AIM++, EvalNet is also integrated.

## Acknowledgement
I would like to extend my heartfelt gratitude to the Deep Learning and Open Source Community, particularly to X, Y, and Z,  whose tutorials and shared wisdom have been a big part of my self-education in computer science and deep learning. This work would not exist without these open and free resources.  
