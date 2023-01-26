# Vision algorithm - SEMS

This repository aim to run a script ([main.py](main.py)) to process n videos/streams.
- Algorithms implemented:
  - People Counter
  - People Tracker
  - 2D Distance Violation
- Uses Flask to publish the image post-processed into a web-page to localhost:8080. 
- Communicates directly with SEMS backend to request the source of the videos/streams and to publish all vision data calculated.

It also aims to run a script using ros (roslaunch sems_vision [covid19_measures.py](catkin_ws/src/sems_vision/scripts/covid19_measures.py)) to process the video stream of a zed2 camera.
- Algorithms implemented:
  - Mask Usage
  - 3D Distance Violation

The code is tested using Ubuntu 18.

## System Requirements
- python3
- python-pip
- Pipenv

## Python Requirements

### Pipenv
- #### Create a new python 3.10 environment
```bash
pipenv --python 3.10
```

- #### Install dependencies
```bash
pipenv install
```

- #### Run the script
```bash
pipenv run python main.py
```

### Dependencies
Instead of managing dependencies through pip, install them using pipenv. This will ensure 
the dependencies are correctly added to the Pipfile.

```bash
pipenv install (DEPEDENCY)
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
- yoloV3.
- Python Dependencies.
- Videos Folder.

Run main.py inside the pipenv environment

Open localhost:8080, all cameras should be displayed over there.
