from multiprocessing import Process, Array
import threading
import numpy as np
import cv2
import imutils
import ctypes

class Camara:
    def __init__(self, inputSource, shared_array, frameShape):
        self.cap = cv2.VideoCapture(inputSource)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        
        self.shared_frame = np.frombuffer(shared_array, dtype=np.uint16)
        self.shared_frame = self.shared_frame.reshape(frameShape)
        
        genFrameTread = threading.Thread(target=self.captureVideo, args=())
        genFrameTread.start()
        genFrameTread.join()

    def captureVideo(self):
        cap = self.cap
                
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            frameResize = imutils.resize(frame, width=500)
            
            if ret:
                self.shared_frame[:] = frameResize
            else:
                break

        # When everything done, release the capture
        cap.release()
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

        camarasArray.append(Array(ctypes.c_uint16, frameShape[0] * frameShape[1] * frameShape[2], lock=False))
        
        processReference.append(Process(target=Camara, args=(inputSource, camarasArray[-1], camarasShape[-1])))
        processReference[-1].start()
    
    while True:
        shared_frame = np.frombuffer(camarasArray[0], dtype=np.uint16)
        shared_frame = shared_frame.reshape(camarasShape[0])
        
        # # Display the resulting frame
        cv2.imshow('frame',shared_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # p.join()
    # q.join()

    # arr = Array('i', range(10))

    # p = Process(target=f, args=(num, arr))
    # p.start()
    # p.join()

    # print(num.value)
    # print(arr[:])