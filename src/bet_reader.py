import cv2
import pytesseract
import logging
from utils import preprocess_bet_area
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure logging
logging.basicConfig(level=logging.INFO)

class BetReader:
    def __init__(self):
        # Tesseract configuration
        self.tess_config = '--psm 7 -c tessedit_char_whitelist=0123456789'

    def read_bets(self, frame, betting_positions):
        bets = []
        frame_height, frame_width = frame.shape[:2]  # Get the frame dimensions
        for pos in betting_positions:
            x, y, r = pos['x'], pos['y'], pos['r']
            logging.info(f"Processing bet position: x={x}, y={y}, radius={r}")

            # Define ROI around the betting circle
            roi_size = int(r * 1.5)
            x_start = max(0, min(x - roi_size, frame_width))
            y_start = max(0, min(y - roi_size, frame_height))
            x_end = max(0, min(x + roi_size, frame_width))
            y_end = max(0, min(y + roi_size, frame_height))

            # Draw a red rectangle around the ROI
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

            # Draw a red circle around the center of the bet position
            cv2.circle(frame, (x, y), r, (0, 0, 255), 2)

            # Check if ROI has valid dimensions
            if x_end > x_start and y_end > y_start:
                bet_roi = frame[y_start:y_end, x_start:x_end]

                # Show the frame with the red box and circle
                cv2.imshow('Bet ROI', frame)
                logging.info(f"Showing ROI for bet position: (x={x}, y={y})")

                # Preprocess and OCR
                preprocessed_bet = preprocess_bet_area(bet_roi)
                bet_amount = self.ocr_bet_amount(preprocessed_bet)

                # Log the bet amount
                logging.info(f"Extracted bet amount: {bet_amount}")
                print(f"Bet Amount at position (x={x}, y={y}): {bet_amount}")

                # Display the preprocessed image used for OCR
                cv2.imshow('Preprocessed Bet', preprocessed_bet)
                cv2.waitKey(500)  # Adjust the wait time for better visualization (500ms)

            else:
                logging.warning(f"Invalid ROI dimensions for bet position: (x={x}, y={y})")

        return bets

    def ocr_bet_amount(self, image):
        text = pytesseract.image_to_string(image, config=self.tess_config)
        # Log the raw OCR text
        logging.info(f"OCR raw text: {text}")

        # Clean and validate the extracted text
        bet_amount = ''.join(filter(str.isdigit, text))
        return int(bet_amount) if bet_amount.isdigit() else 0

# Ensure OpenCV windows close properly when the program ends
cv2.destroyAllWindows()
