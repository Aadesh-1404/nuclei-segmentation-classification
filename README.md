# Nuclei-Segmentation-Classification-in-Histopathology-Datasets-Summer-Intern

This repository includes a brief report of the project completed during the summer internship at MeDAL IIT Bombay under Prof.Amit Sethi.

# Introduction

In cancer pathology, the presence and mix of  immune cells that are seen in and around a tumour can be linked to how dangerous a particular cancer is, and whether it can be treated with the planned treatment. To study this, the first step is to detect and segment different types of cells in digital pathology images. I was provided two such datasets and the task of the project was to:
1) Develop a solution for instance segmentation on the MoNuSeg dataset.
2) Develop a solution for instance segmentation and cell-type classification on the MoNuSAC dataset.

# Building Nuclei Segmentation Methods

**Mask-RCNN, Hover-Net, U-Net, PatchEUNet** were studied and implemented. 
- The file ```MaskRCNN_Scratch.ipynb``` contains a Mask-RCNN code in Pytorch build from scratch. The model built has some faults needs some minor changes for an entire dataset to train.
