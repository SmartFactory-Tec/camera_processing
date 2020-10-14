# ContadorPersonas
To run the code simply create a venv outside the repository
Activate it

Install the requirements
  $ pip install -r requirements.txt

Try testing with the example_01.mp4 video

$ python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
--input videos/peoplewalking.mp4 --output output/output_01.avi
