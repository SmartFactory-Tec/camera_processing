# Vision algorithm - SEMS

This repository aim to run a script (main.py) to process n videos/streams.
- Algorithms implemented:
  - People Counter
  - People Tracker
  - 2D Distance Violation
- Uses Flask to publish the image post-processed into a web-page to localhost:8080. 
- Communicates directly with SEMS backend to request the source of the videos/streams and to publish all vision data calculated.

It also aims to run a script using ros (roslaunch sems_vision covid19_measures.py) to process the video stream of a zed2 camera.
- Algorithms implemented:
  - Mask Usage
  - 3D Distance Violation

The code is tested using Ubuntu 18.

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

## Usage script
Open main.py, modify global variables if needed.

```python
CAMARAIDS = [6, 7]
BACK_ENDPOINT = ["http://sems.back.ngrok.io/", "http://localhost:3001/"][0]
NGROK_AVAILABLE = True
GPU_AVAILABLE = True
VERBOSE = False
CONFIDENCE_ = 0.3
SKIP_FRAMES_ = 25
```

Requirements:
- Valid CamaraIDS.
- Backend running.
- Yolov3.
- Python Dependencies.
- Videos Folder.

Run main.py

Open localhost:8080, all camaras should be displayed overthere.
