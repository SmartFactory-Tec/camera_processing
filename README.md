# ContadorPersonas

This branch aim to run a script to process a video/stream and count the people in every frame.
The code is tested using Ubuntu 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements after you have activated your venv created in somewhere else.

```bash
pip install -r requirements.txt
```

## Usage

To run the people_counter.py script run the following line, giving the protxt, caffemodel, input video and output destination as arguments.
```bash
python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \

--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \

--input videos/peoplewalking.mp4 --output output/output_01.avi
```
Then, you should see a window displaying the labeled video and an output in console similar to this

```bash
[INFO] loading model...
[INFO] opening video file...
[INFO] elapsed time: 20.95
[INFO] approx. FPS: 61.25
```