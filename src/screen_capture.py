# blackjack_analyzer/screen_capture.py

import mss
import cv2
import numpy as np

class ScreenCapture:
    def __init__(self, monitor_number=1):
        self.monitor_number = monitor_number
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[self.monitor_number]

    def capture_screen(self):
        while True:
            img = self.sct.grab(self.monitor)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            yield frame

    def release(self):
        self.sct.close()