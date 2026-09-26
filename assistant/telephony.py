"""Offline transport only. This module cannot place a telephone call."""


class SimulatedTransport:
    def __init__(self):
        self.connected = False

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False
