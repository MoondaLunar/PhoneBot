"""Bounded asynchronous event coordinator for a single simulated session.

This is deliberately transport-agnostic. Only trusted callers may submit events.
The existing Session authorization and speech catalog remain authoritative.
"""
import asyncio
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

from .state import State


class EventKind(str, Enum):
    REMOTE = "remote"
    RESPOND = "respond"
    ABORT = "abort"
    DISCONNECT = "disconnect"


@dataclass(frozen=True)
class Event:
    kind: EventKind
    text: str = ""
    remote_kind: str = ""
    message_id: str = ""
    question_id: str = ""
    pending_id: str = ""


class QueueFullError(RuntimeError):
    pass


class AsyncSessionController:
    """Serializes events without allowing a stale worker to continue after shutdown.

    Blocking model adapters execute in a thread. Cancellation cannot forcibly stop
    their underlying network requests, but it prevents late results from changing
    the active session after shutdown. Do not use this with live audio until the
    transport has its own output cancellation and disconnect guarantees.
    """

    def __init__(self, session, *, max_events=128, max_pending=16, event_timeout=30.0):
        if not 1 <= max_events <= 10000 or not 1 <= max_pending <= 1000:
            raise ValueError("Invalid event limits")
        if not 0 < event_timeout <= 120:
            raise ValueError("Invalid event timeout")
        self.session = session
        self.max_events = max_events
        self.event_timeout = event_timeout
        self.queue = asyncio.Queue(maxsize=max_pending)
        self.generation = uuid4().hex
        self.closed = False
        self._processed = 0
        self._worker = None
        self._inflight = None
        self._lock = asyncio.Lock()

    async def submit(self, event: Event):
        if type(event) is not Event or type(event.kind) is not EventKind:
            raise TypeError("Expected a typed event")
        if event.kind in {EventKind.ABORT, EventKind.DISCONNECT}:
            await self.shutdown(disconnect=event.kind is EventKind.DISCONNECT)
            return
        if self.closed or self.session.state == State.ENDED:
            raise RuntimeError("Controller closed")
        if self._processed + self.queue.qsize() >= self.max_events:
            raise QueueFullError("Session event limit reached")
        try:
            self.queue.put_nowait((self.generation, event))
        except asyncio.QueueFull:
            raise QueueFullError("Pending event queue full") from None

    async def run(self):
        if self._worker is not None:
            raise RuntimeError("Controller already running")
        self._worker = asyncio.current_task()
        try:
            while not self.closed and self.session.state != State.ENDED:
                generation, event = await self.queue.get()
                try:
                    if generation != self.generation or self.closed:
                        continue
                    self._processed += 1
                    async with self._lock:
                        if self.closed or generation != self.generation:
                            continue
                        self._inflight = asyncio.create_task(asyncio.to_thread(self._apply, event))
                        try:
                            await asyncio.wait_for(self._inflight, timeout=self.event_timeout)
                        except asyncio.TimeoutError:
                            await self.shutdown(disconnect=True)
                            break
                        finally:
                            self._inflight = None
                finally:
                    self.queue.task_done()
        except asyncio.CancelledError:
            raise
        finally:
            self.closed = True

    def _apply(self, event):
        if event.kind is EventKind.REMOTE:
            self.session.remote(event.remote_kind, event.text,
                                message_id=event.message_id, question_id=event.question_id)
        elif event.kind is EventKind.RESPOND:
            self.session.respond(event.text, pending_id=event.pending_id)
        else:
            raise ValueError("Unsupported event")

    async def shutdown(self, *, disconnect=False):
        if self.closed:
            return
        self.closed = True
        self.generation = uuid4().hex
        if self._inflight is not None:
            self._inflight.cancel()
        # Session may be processing on a worker thread. Wait for it to exit before
        # changing state, so an old result cannot write into a closed session.
        async with self._lock:
            if self.session.state != State.ENDED:
                if disconnect:
                    self.session.disconnect()
                else:
                    self.session.abort()
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
