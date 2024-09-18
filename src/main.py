# blackjack_analyzer/main.py

import threading
from screen_capture import ScreenCapture
from betting_circle_detector import BettingCircleDetector
from card_recognizer import CardRecognizer
from bet_reader import BetReader
from state_manager import StateManager
import cv2

class BlackjackAnalyzer:
    def __init__(self):
        self.screen_capture = ScreenCapture()
        self.betting_circle_detector = BettingCircleDetector()
        self.card_recognizer = CardRecognizer()
        self.bet_reader = BetReader()
        self.state_manager = StateManager()
        self.running = True

    def run(self):
        for frame in self.screen_capture.capture_screen():
            # Detect betting circles
            betting_positions = self.betting_circle_detector.detect(frame)
            # Read bets
            bets = self.bet_reader.read_bets(frame, betting_positions)
            # Detect and recognize cards
            card_positions = self.card_recognizer.detect_cards(frame)
            recognized_cards = self.card_recognizer.recognize_cards(frame, card_positions)
            # Update state
            self.state_manager.update(recognized_cards)
            # Display or log data
            self.display_data(bets, recognized_cards)
            # Exit condition
            if self.exit_condition():
                break
        self.cleanup()

    def display_data(self, bets, cards):
        # Implement display or logging functionality
        print("Bets:", bets)
        print("Recognized Cards:", cards)
        # For visualization (optional)
        # cv2.imshow('Blackjack Analyzer', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     self.running = False

    def exit_condition(self):
        # Define exit condition (e.g., pressing 'q')
        return not self.running

    def cleanup(self):
        # Release resources
        cv2.destroyAllWindows()
        self.screen_capture.release()

if __name__ == "__main__":
    analyzer = BlackjackAnalyzer()
    analyzer.run()