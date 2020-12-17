from multiprocessing import Process, Array
import threading
import numpy as np
import cv2
import time
import imutils
import ctypes
from imutils.video import FPS

class Camara:
    def __init__(self, inputSource, shared_array, frameShape):
        self.cap = cv2.VideoCapture(inputSource)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        
        shared_frame = np.frombuffer(shared_array, dtype=np.uint8)
        shared_frame = shared_frame.reshape(frameShape)
        self.captureVideo(shared_frame)
        

    def captureVideo(self, shared_frame):
        while(True):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            
            frameResize = imutils.resize(frame, width=500)
            
            if ret:
                shared_frame[:] = frameResize
            else:
                break

            # Display the resulting frame
            cv2.imshow('frame',frameResize)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    # captureVideo()
    processReference = []
    camarasShape = []
    camarasArray = []
    inputSources = ["rtsp://10.13.118.14/stream1"] #,"rtsp://visitante1:visitante1@10.14.244.16/stream1"]
    
    for inputSource in inputSources: 
        cap = cv2.VideoCapture(inputSource)
        ret, frame = cap.read()
        
        frame = imutils.resize(frame, width=500)
        frameShape = frame.shape
        camarasShape.append(frameShape)
        cap.release()
        
        camarasArray.append(Array(ctypes.c_uint8, frameShape[0] * frameShape[1] * frameShape[2], lock=False))
        # camarasArray.append(Array('i', range(10)))
        
        processReference.append(Process(target=Camara, args=(inputSource, camarasArray[-1], camarasShape[-1])))
        processReference[-1].start()
    
    fps = FPS().start()
    while True:
        # print(camarasArray[0][:])
        shared_frame = np.frombuffer(camarasArray[0], dtype=np.uint8)
        shared_frame = shared_frame.reshape(camarasShape[0])
        
        # # Display the resulting frame
        cv2.imshow('frame',shared_frame)
        
        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1/60)
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cv2.destroyAllWindows()
            
    # p.join()
    # q.join()

    # arr = Array('i', range(10))

    # p = Process(target=f, args=(num, arr))
    # p.start()
    # p.join()

    # print(num.value)
    # print(arr[:])