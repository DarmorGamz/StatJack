# blackjack_analyzer/betting_circle_detector.py

import cv2
import numpy as np
from utils import resize_frame

class BettingCircleDetector:
    def __init__(self):
        # Parameters for circle detection
        self.dp = 1.2
        self.min_dist = 50
        self.param1 = 50
        self.param2 = 30
        self.min_radius = 20
        self.max_radius = 50

    def detect(self, frame):
        # Optionally resize frame
        # frame = resize_frame(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        betting_positions = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                betting_positions.append({'x': x, 'y': y, 'r': r})
                # For visualization (optional)
                # cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        return betting_positions