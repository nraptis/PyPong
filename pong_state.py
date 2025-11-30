# pong_state.py

from enum import Enum, auto

class PongState(Enum):
    Idle = auto()
    Playing = auto()
    Dead = auto()