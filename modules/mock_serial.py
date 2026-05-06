"""Mock Serial port for offline simulation and testing.

Provides a Serial-like object implementing `write`, `readline`, `flush`,
`close` and `in_waiting` property. It responds to common commands used by
the firmware so the Python stack can be exercised without actual hardware.
"""
import time
import threading
from collections import deque


class MockSerial:
    def __init__(self, response_delay: float = 0.05):
        self._open = True
        self._lock = threading.Lock()
        self._out = deque()
        self._response_delay = float(response_delay)

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def in_waiting(self):
        with self._lock:
            return len(self._out) > 0

    def write(self, data: bytes):
        s = data.decode('utf-8', errors='ignore').strip()
        # spawn a responder thread to simulate firmware reply
        threading.Thread(target=self._respond, args=(s,), daemon=True).start()

    def flush(self):
        return

    def readline(self) -> bytes:
        # wait until a response appears (bounded sleep)
        start = time.time()
        while time.time() - start < 5.0:
            with self._lock:
                if self._out:
                    return (self._out.popleft() + "\n").encode('utf-8')
            time.sleep(0.01)
        return b""

    def close(self):
        self._open = False

    # internal responder
    def _respond(self, cmd: str):
        # small processing delay
        time.sleep(self._response_delay)
        up = cmd.strip().upper()
        resp = None

        if up.startswith("PING"):
            resp = "PONG"
        elif up.startswith("TEST"):
            resp = "OK TEST"
        elif up.startswith("INIT"):
            resp = "ACK INIT"
        elif up.startswith("HOME"):
            resp = "ACK HOME"
        elif up.startswith("MOVE"):
            resp = f"ACK MOVE {up.split()[1] if len(up.split())>1 else ''}".strip()
        elif up.startswith("PICK"):
            resp = "ACK PICK"
        elif up.startswith("PLACE"):
            # accept both PLACE idx and PLACE r c
            resp = "ACK PLACE"
        elif up.startswith("DEMO"):
            resp = "ACK DEMO"
        elif up.startswith("PWMTEST"):
            resp = "OK PWMTEST"
        elif up.startswith("STATUS"):
            resp = "STATUS OK"
        else:
            resp = "OK"

        with self._lock:
            # mimic some intermittent debug lines
            self._out.append(f"BOOT: simulated firmware")
            self._out.append(resp)
