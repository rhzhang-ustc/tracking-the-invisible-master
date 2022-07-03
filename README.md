# tracking-the-invisible-master

Implementation for 2010 CVPR 'tracking-the-invisible'


## Environment

opencv-python < 3.4.2.16 to use sift features

Python 3.6

## Usage

For simplicity, just run `python main.py` to generate the demo video (test.mp4). The default setting is to track claw of the robot. 

If you want to try some other videos:

```
python main.py --video_path data/visuo_test.mp4 --object_location (288, 208) --output_path test.mp4
```




