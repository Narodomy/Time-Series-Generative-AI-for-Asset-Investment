from typing import Dict, List

class Portfolio:
    def __init__(self, name: str, weights: Dict[str, float]):
        """weights: such as {'AAPL': 0.5, 'TSLA': 0.5}"""
        self.name = name
        self.weights = weights
        self.holdings = list(weights.keys())