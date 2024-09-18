# blackjack_analyzer/utils.py

import cv2
import os

def resize_frame(frame, width=None, height=None):
    # Resize frame while maintaining aspect ratio
    if width is not None:
        r = width / float(frame.shape[1])
        dim = (width, int(frame.shape[0] * r))
    elif height is not None:
        r = height / float(frame.shape[0])
        dim = (int(frame.shape[1] * r), height)
    else:
        return frame
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized

def preprocess_card(card_image):
    # Preprocess card image for template matching
    gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 300))
    return resized

def load_card_templates():
    # Load card templates from a directory
    templates = {}
    template_dir = 'card_templates/'
    for filename in os.listdir(template_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            card_name = os.path.splitext(filename)[0]
            template_image = cv2.imread(os.path.join(template_dir, filename), 0)
            templates[card_name] = cv2.resize(template_image, (200, 300))
    return templates

def preprocess_bet_area(bet_roi):
    # Preprocess bet area for OCR
    gray = cv2.cvtColor(bet_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh