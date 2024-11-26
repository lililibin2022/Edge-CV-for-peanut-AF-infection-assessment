# Edge-CV-for-peanut-AF-infection-assessment
Assessment of A. flavus infection indices

## Requirements

Python 3.6+
YOLOv11

```bash
pip install ultralytics 
```
## Installation

## Results

### Results of segmentaiton

![image](https://github.com/user-attachments/assets/67e652b8-4950-4a88-ad01-4b4452098505)

### Results of post-processing

![image](https://github.com/user-attachments/assets/81719b0f-0c73-4263-9174-15de95646865)


## Detection in Videos

- Create a folder with name `videos` in the same directory
- Dump your videos in this folder
- In `settings.py` edit the following lines.

```python
# video
VIDEO_DIR = ROOT / 'videos' # After creating the videos folder

### Detection on RTSP

- Select the RTSP stream button
- Enter the rtsp url inside the textbox and hit `Detect Objects` button

### Detection on YouTube Video URL

- Select the source as YouTube
- Copy paste the url inside the text box.
- The detection/segmentation task will start on the YouTube video url

## Acknowledgements

This app is based on the YOLOv11(<https://github.com/ultralytics/ultralytics>) segmentation algorithm. THANKS!!!

### Disclaimer

Please note that this project is intended for educational purposes only and should not be used in production environments.



