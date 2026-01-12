from .basket import Basket

class Window:
    def __init__(size):
        self.size = size

class Market:
    def __init__(self, basket: Basket, window: Window, stride: int = 1):
        self.basket = basket
        self.window = window
        self.stride = stride