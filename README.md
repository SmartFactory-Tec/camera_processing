# Vision algorithm - People Counter

This branch aim to run a script to process video/stream and count the people in every frame. V2.0 and V3.0 output could be seen on a web page. Script version 3.0 was adapted to process n video/streams at the same time.

The code is tested using Ubuntu.

## System Requirements
- python3
- python-pip
- virtualenv (Recommended)

## Python Requirements

### Virtualenv
- #### Create
> If you haven't create a virtualenv, create one.
```bash
virtualenv -p python3 _NAME_
```

- #### Activate
> Before running the script active the virtualenv.
```bash
source _NAME_/bin/activate
```

- #### Deactivate
> If you are done running the script deactive the virtualenv.
```bash
deactivate
```

### Dependencies

> Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements after you have activated your venv created in somewhere else.

```bash
pip install -r requirements.txt
```

## Usage script v1.0

Run people_counter_v1.py script, giving the protxt, caffemodel, input video and output destination as arguments.
```bash
python people_counter_v1.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/peoplewalking.mp4 --output output/output_01.avi
```
Then, you should see a window displaying the labeled video and an output in console similar to this

```bash
[INFO] loading model...
[INFO] opening video file...
[INFO] elapsed time: 20.95
[INFO] approx. FPS: 61.25
```

## Usage script v2.0

Run people_counter_v2.py script, giving the protxt, caffemodel, input video/stream.
```bash
python people_counter_v2.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/peoplewalking.mp4
```
Then, you should open localhost:8080 in your browser and be able to see the video processing.

## Usage script v3.0
Build up inputScript.json according to your input, follow up this format:
```json
{
  "idLocation0" : {
    "name" : "name_location0",
    "description" : "description0",
    "camaras" : [
      {
        "name" : "name_camara0", 
        "src" : "source_camara0"
      },
      {
        "name" : "name_camara1", 
        "src" : "source_camara1"
      }
    ]
  },
  "idLocation1" : {
    "name" : "name_location1",
    "description" : "description1",
    "camaras" : [
      {
        "name" : "name_camara", 
        "src" : "source_camara"
      },
    ]
  }
}
```

Run people_counter_v3.py script.
```bash
python people_counter_v3.py
```

Then, you should open localhost:8080 in your browser and be able to see the video processing. One tab per each location and one carousel per each camara in the same location.