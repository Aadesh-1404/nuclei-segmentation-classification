# Nuclei-Segmentation-Classification-in-Histopathology-Datasets-Summer-Intern

This repository includes a brief report of the project completed during the summer internship at MeDAL IIT Bombay under Prof.Amit Sethi.

# Introduction

In cancer pathology, the presence and mix of  immune cells that are seen in and around a tumour can be linked to how dangerous a particular cancer is, and whether it can be treated with the planned treatment. To study this, the first step is to detect and segment different types of cells in digital pathology images. I was provided two such datasets and the task of the project was to:
1) Develop a solution for instance segmentation on the MoNuSeg dataset.
2) Develop a solution for instance segmentation and cell-type classification on the MoNuSAC dataset.

# Building Nuclei Segmentation Methods

**Mask-RCNN, Hover-Net, U-Net, PatchEUNet** were studied and implemented. 
- The file ```MaskRCNN_Scratch.ipynb``` contains a Mask-RCNN code in Pytorch build from scratch. The model built has some faults needs some minor changes for an entire dataset to train.
- The file ```HoVerNet_Pytorch_from_scratch.ipynb``` contains a Hover-Net code in Pytorch build from scratch. The model built can be trained on a datset but does not contain the postprocessing and inference steps, adding this steps cam provide results on any dataset.
- The file ```Unet_MoNuSeg.ipynb``` contains a complete-tested U-Net model built in Keras-tensorflow.
- PatchEUnet model was built using **Segmentation Models Python API**.

# MoNuSeg Dataset

The Dataset contains:
- Annotated tissue images of several patients with tumors of different organs, diagnosed at multiple hospitals.
- H&E stained tissue images captured at 40x magnification from [TCGA](http://cancergenome.nih.gov/) archive. 

[Training data](https://drive.google.com/file/d/1JZN9Jq9km0rZNiYNEukE_8f0CsSK3Pe4/view?usp=sharing): 30 images & around 22,000 nuclear boundary annotations.

[Test data](https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw/view?usp=sharing): 14 images & around 7000 nuclear boundary annotations.
Images are in .tif format and boundary annotations are  .xml files.

## Experiment 1(a):-

I first analysed the Mask-RCNN model as suggested in the [paper](https://arxiv.org/abs/1703.06870). Used [TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) to segment nuclei in MoNuSeg dataset. Used the pretrained(COCO weights)  torchvision.models.detection.maskrcnn_resnet50_fpn model and fine tuned the three branches on the MoNuSeg dataset. Dice + binary cross entropy was used as the Loss function of Mask branch.

### Conclusion:

Got good predictions by using the fine tuned weights. The confidence score > 0.5 was used to detect the nuclei in the test images.
The results on one of the images is shown below:

![TCGA-IZ-8196-01A-01-BS1](https://user-images.githubusercontent.com/68186100/128328052-b09556f8-2d5f-420f-adc9-d11e036e5394.png)

![Fine tune](https://user-images.githubusercontent.com/68186100/128327748-bef87a93-e899-4ce3-9811-3308693ab5b0.png)





