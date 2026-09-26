from enum import Enum


class State(str, Enum):
    READY = "READY"
    DIALING = "DIALING"
    CONNECTED = "CONNECTED"
    AGENT_SPEAKING = "AGENT_SPEAKING"
    LISTENING = "LISTENING"
    ASK_USER = "ASK_USER"
    ENDING = "ENDING"
    ENDED = "ENDED"
    ERROR = "ERROR"


TRANSITIONS = {
    State.READY: {State.DIALING, State.ENDING},
    State.DIALING: {State.CONNECTED, State.ENDING, State.ERROR},
    State.CONNECTED: {State.AGENT_SPEAKING, State.ENDING, State.ERROR},
    State.AGENT_SPEAKING: {State.LISTENING, State.ASK_USER, State.ENDING, State.ERROR},
    State.LISTENING: {State.AGENT_SPEAKING, State.ASK_USER, State.ENDING, State.ERROR},
    State.ASK_USER: {State.AGENT_SPEAKING, State.ENDING, State.ERROR},
    State.ENDING: {State.ENDED, State.ERROR},
    State.ERROR: {State.ENDING},
    State.ENDED: set(),
}


class Machine:
    def __init__(self):
        self.state = State.READY

    def move(self, target):
        if not isinstance(target, State) or target not in TRANSITIONS[self.state]:
            raise ValueError(f"Invalid transition from {self.state.value}")
        self.state = target
