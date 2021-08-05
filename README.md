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
- Annotated tissue images of 30 patients with tumors of different organs, diagnosed at multiple hospitals.
- H&E stained tissue images captured at 40x magnification from [TCGA](http://cancergenome.nih.gov/) archive. 

[Training data](https://drive.google.com/file/d/1JZN9Jq9km0rZNiYNEukE_8f0CsSK3Pe4/view?usp=sharing): 30 images & around 22,000 nuclear boundary annotations.

[Test data](https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw/view?usp=sharing): 14 images & around 7000 nuclear boundary annotations.
Images are in .tif format and boundary annotations are  .xml files.

## Experiment 1(a):-

I first analysed the Mask-RCNN model as suggested in the [paper](https://arxiv.org/abs/1703.06870). Used [TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) to segment nuclei in MoNuSeg dataset. Firstly, Used [XML_to_Binary](https://drive.google.com/file/d/15oAnCnPchvn40U4h_mwgoTArlpBs2gK3/view?usp=sharing) code for reading xml annotations and creating binary masks and then created Non-Overlapping patches of images & masks of size 250x250 from original size of 1000x1000. Used the pretrained(COCO weights)  torchvision.models.detection.maskrcnn_resnet50_fpn model and fine tuned the three branches on the MoNuSeg dataset. Dice + binary cross entropy was used as the Loss function of Mask branch. Also, used helper functions like augmentation before training.

### Conclusion:

Got good predictions by using the [fine tuned weights](https://drive.google.com/file/d/1vI48TkKg-0KSmatR2nNDmEf_d_8swGcp/view?usp=sharing). The confidence score > 0.5 was used to detect the nuclei in the test images.
The results on one of the images is shown below: The model was able to detect 263 nuclei.

Original Test Image            |  Predicted Masks
:-------------------------:|:-------------------------:
<img src=https://user-images.githubusercontent.com/68186100/128328052-b09556f8-2d5f-420f-adc9-d11e036e5394.png width="500" height="500"> |<img src=https://user-images.githubusercontent.com/68186100/128327748-bef87a93-e899-4ce3-9811-3308693ab5b0.png width="500" height="500">

The file ```Maskrcnn_pytorch_FineTuning.ipynb``` in the folder ```MoNuSeg- Exp1(a)``` contains the code.

## Experiment 1(b):-

To experiment the Mask-RCNN model with different loss functions [Matterport Mask-RCNN](https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py) was used.
Loss funtions like **Binary Cross-Entropy loss , Focal loss & Binary + Dice loss** were tried out. 

First, Read the xml annotations and created masks of individual nuclei and saved it as .png format using the [code](https://drive.google.com/file/d/1wprt3MR8XwrF4xZO5KbUUhW9ZA2gF7FJ/view?usp=sharing). Then, created Non-Overlapping patches of images & masks of size 500x500 from original size of 1000x1000. Also, used additional 2018 Data Science Bowl dataset to increase the length of training dataset. I trained the Mask-RCNN model from scratch and got the results.

The folder ```MoNuSeg- Exp1(b)``` contains all the files for this experiment. The ```mrcnn_matterport.py``` file contains the main parts of the code, which when run trains the model or detects the nuclei using the commands used in ```MRCNN_Matterport_train.ipynb```. 
Two Jupyter notebooks are provided as well:```MRCNN_Data_Visualisation.ipynb``` and ```MRCNN_Inference.ipynb``` explores the dataset, run stats on it, and goes through the detection process step by step.

Here is the the [training](https://drive.google.com/drive/folders/1WW45nYX0MF3f5Akw98cFpAMMhxJaWdJ9?usp=sharing) and [testing](https://drive.google.com/drive/folders/14st29nEtU7CIubNZbo_g84lpdE4GQ41a?usp=sharing) obtained after preprocessing.

### Conclusions:

Training loss and Validation loss decreases with increasing epochs as seen in figure below:

<img src=https://user-images.githubusercontent.com/68186100/128333760-bd388de7-7792-4e83-b017-64ca6d83f44e.png width="500" height="500">

1) The Binary Cross-Entropy loss with COCO pretrained weights gave the best results on many test images, with **mAP @0.5 = 0.5487 and Mean AP(@0.5-0.95) over 5 images = 0.1762**
2) Focal loss with Imagenet pretrained weights provided extremely well results on selected images.

The results were very good. Some of the results can be seen below:

<img src=https://user-images.githubusercontent.com/68186100/128336646-989cf31c-bd50-432c-ae37-bc0e6d00f83c.jpeg>
<img src=https://user-images.githubusercontent.com/68186100/128336838-45d6d125-9289-4d04-95c1-34191ae920e0.jpeg>

More results can be found in ```Results``` folder.

The trained weights can be accessed [here](https://drive.google.com/file/d/10X1ZOT4Aoy4QJ_O_ZMlKbcSK9ouRVTWc/view?usp=sharing).

## Experiment 2:-

The U-Net network can work with very few training images and yield more precise segmentations, hence work would have been incomplete without exploring this model for the Nuclei segmentation task. The code for the network can be found in the file ```Unet_MoNuSeg.ipynb``` in the ```MoNuSeg- Exp2``` folder. The model was trained from scratch on MoNuSeg dataset and tremendous results were obtained. Some of the result are shown below:-

The second image is the Ground truth masks and the third image is the predicted masks.

![exp2](https://user-images.githubusercontent.com/68186100/128338166-a0ba801a-b37d-41f3-b512-29f2b70d2f03.jpeg)



# MoNuSAC Dataset
The dataset contains annotated dataset of H&E stained digitized tissue images from 4 different organs (Lung, Prostate, Kidney, and Breast) of several patients acquired at multiple hospitals using one of the most common 40x scanner magnification. The datasets includes images in .svs & .tif format and nuclear boundary annotations in .xml files.

Here is the [testing](https://drive.google.com/file/d/1G54vsOdxWY1hG7dzmkeK3r0xz9s-heyQ/view?usp=sharing) and [training](https://drive.google.com/file/d/1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq/view) data. 

## PatchEUNet Model

Analysed and Followed the steps suggested in the [Github](https://github.com/hasibzunair/MoNuSAC-ISBI-2020) and trained the model from scratch using MoNuSAC dataset.
First, the H&E stained images(.svs) were used along with associated xml files to generate binary masks of each annotated cell-type using [code](https://github.com/ruchikaverma-iitg/MoNuSAC/blob/master/Binary_mask_generation.ipynb) provided by MoNuSAC’20. Then, combined the binary masks of each annotated cell-type in a single mask and created a folder containing images & the ground truth masks and finally created overlapping patches of images and masks of size 128x128 with step size: 110.

Here is the [training](https://drive.google.com/drive/folders/1nk5iEdMsZGly-3rxmt9vkICXMthbICRP?usp=sharing) dataset finally obtained after preprocessing.

The codes for the model can be found in ```MoNuSAC``` folder: ```MoNuSac_Train.ipynb``` contains all the training part of the model, ```MoNuSac_Data_Preparation.ipynb``` contains the preprocesssing of the datasets, ```MoNuSac_Inference.ipynb``` contains the code for detecting nuclei on test images.

### Conclusions:

Obtained a Mean Panoptic Quality score without filer of **0.448**. The results obtained were good with few false negatives. The trained weights can be accessed from [here](https://drive.google.com/file/d/1-8v-bOZz3bYlVmxjQIBYAw5sz_3s3eXq/view?usp=sharing). Few results are displayed below:-

Here, ‘Background’ = “black",'Epithelial' = "red", 'Lymphocyte' = "yellow", 'Neutrophil' = "blue", 'Macrophage' = "green".

![3](https://user-images.githubusercontent.com/68186100/128340885-11aa431e-82a2-4cae-a1fa-914c45d6c204.jpeg)

Image showing the classification of the 4 cells:

![pred](https://user-images.githubusercontent.com/68186100/128340991-a7e3bd41-fcc0-485d-ae87-7056b2ed6942.png)


## The results obtained with all the models on both the datasets were good enough but there can be futher improvement in results by using Other Nuclei Segmentation Methods/ Models, experimenting various Loss functions and also tuning of Hyperparameters.

# Acknowledgement 
Prof. Amit Sethi for providing this opportunity to explore the field of medical imaging segmentation using Computer Vision, Machine Learning.



