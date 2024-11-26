# Edge-CV-for-peanut-AF-infection-assessment
Assessment of A. flavus infection indices

## Requirements
ultralytics
Python 3.11
```bash
from ultralytics import YOLO
import torch
from pathlib import Path
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as sk_label
from skimage.segmentation import watershed
from torchviz import make_dot
import netron
from torchsummary import summary   

if __name__ == "__main__":
    choice = input("Enter 'train' to train the model, 'analyze' to process images or "Visualize" to visualize instances with results: ").strip().lower()
    if choice == 'train':
        train_model()
    elif choice == 'analyze':
        analyze_images()
    elif choice == 'Visualize':
        visualize_instances_with_colors()


```
## Model architecture
https://netron.app/
## Results

### Results of segmentaiton

![image](https://github.com/user-attachments/assets/67e652b8-4950-4a88-ad01-4b4452098505)

### Results of post-processing

![image](https://github.com/user-attachments/assets/81719b0f-0c73-4263-9174-15de95646865)


## Detection in Videos

- Create a folder with name `videos` in the same directory
- Dump your videos in this folder
- In `settings.py` edit the following lines.


## Acknowledgements

This app is based on the YOLOv11(<https://github.com/ultralytics/ultralytics>) segmentation algorithm. THANKS!!!

### Disclaimer

Please note that this project is intended for educational purposes only and should not be used in production environments.



