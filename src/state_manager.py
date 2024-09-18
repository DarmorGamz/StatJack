# blackjack_analyzer/state_manager.py

class StateManager:
    def __init__(self):
        self.previous_cards = set()
        self.current_cards = set()

    def update(self, recognized_cards):
        self.current_cards = set([
            (card['position'], card['card']) for card in recognized_cards
        ])
        # Identify new cards
        new_cards = self.current_cards - self.previous_cards
        if new_cards:
            # Process new cards
            for card in new_cards:
                print(f"New card detected: {card[1]} at position {card[0]}")
        # Update previous cards for the next iteration
        self.previous_cards = self.current_cards.copy()
        # Detect if hand has ended (e.g., no cards on the table)
        if not self.current_cards:
            self.reset()

    def reset(self):
        # Reset the state when a hand ends
        self.previous_cards.clear()
        print("Hand has ended. State has been reset.")