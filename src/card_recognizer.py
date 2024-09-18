# blackjack_analyzer/card_recognizer.py

import cv2
import numpy as np
from utils import preprocess_card, load_card_templates

class CardRecognizer:
    def __init__(self):
        # Load card templates for template matching
        self.card_templates = load_card_templates()

    def detect_cards(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        card_contours = []
        for cnt in contours:
            approx = cv2.approxPolyDP(
                cnt, 0.01 * cv2.arcLength(cnt, True), True
            )
            if len(approx) == 4 and cv2.contourArea(cnt) > 1000:
                card_contours.append(approx)
        return card_contours

    def recognize_cards(self, frame, card_contours):
        recognized_cards = []
        for cnt in card_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            card_roi = frame[y:y+h, x:x+w]
            preprocessed_card = preprocess_card(card_roi)
            best_match = self.match_card(preprocessed_card)
            if best_match:
                recognized_cards.append({
                    'position': (x, y, w, h),
                    'card': best_match
                })
        return recognized_cards

    def match_card(self, card_image):
        # Implement template matching
        best_score = None
        best_match = None
        for card_name, template in self.card_templates.items():
            res = cv2.matchTemplate(card_image, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if best_score is None or max_val > best_score:
                best_score = max_val
                best_match = card_name
        # Define a threshold for a valid match
        if best_score and best_score > 0.7:
            return best_match
        else:
            return None