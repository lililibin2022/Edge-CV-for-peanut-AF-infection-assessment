# Edge-CV-for-peanut-AF-infection-assessment
Assessment of A. flavus infection indices

## Requirements

Python 3.11
YOLOv11
    "numpy>=1.23.0",
    "numpy<2.0.0; sys_platform == 'darwin'", # macOS OpenVINO errors https://github.com/ultralytics/ultralytics/pull/17221
    "matplotlib>=3.3.0",
    "opencv-python>=4.6.0",
    "pillow>=7.1.2",
    "pyyaml>=5.3.1",
    "requests>=2.23.0",
    "scipy>=1.4.1",
    "torch>=1.8.0",
    "torch>=1.8.0,!=2.4.0; sys_platform == 'win32'", # Windows CPU errors w/ 2.4.0 https://github.com/ultralytics/ultralytics/issues/15049
    "torchvision>=0.9.0",
    "tqdm>=4.64.0", # progress bars
    "psutil", # system utilization
    "py-cpuinfo", # display CPU info
    "pandas>=1.1.4",
    "seaborn>=0.11.0", # plotting
    "ultralytics-thop>=2.0.0", # FLOPs computation https://github.com/ultralytics/thop

```bash
pip install ultralytics 
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



