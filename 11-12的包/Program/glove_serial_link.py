"""Wired glove COM port that reopens itself when the glove stops sending.

On 2026-11-06 the right glove stopped sending three times while its COM port stayed open; the
data only came back after the program was restarted. ReconnectingSerial behaves like
serial.Serial for the programs (.port, .is_open, .readline(), .close()) and:
  * a read error (USB unplugged / re-enumerated) closes the port and retries opening it every
    RETRY_S seconds;
  * no line for SILENT_REOPEN_S seconds closes and reopens the port (opening the port resets the
    XIAO ESP32C6, so a stuck glove restarts; the JY901 keeps its power and its angles).
Every reopen is counted in stats(); the take's own silence check still marks the gap as unusable.
"""
from __future__ import annotations

import time
from collections import deque

import serial

BAUD = 460800
SILENT_REOPEN_S = 1.0
RETRY_S = 1.0


class ReconnectingSerial:
    def __init__(self, port: str, baudrate: int = BAUD, timeout: float = 0.2, opener=None, clock=time.monotonic):
        self.port = port; self.baudrate = baudrate; self.timeout = timeout
        self.opener = opener or (lambda: serial.Serial(port, baudrate, timeout=timeout))
        self.clock = clock; self.is_open = True
        self.reopens = 0; self.errors = 0; self.silent_reopens = 0; self.events: deque = deque(maxlen=200)
        self.ser = self.opener()          # the first open raises, like serial.Serial
        self.last_line = self.clock(); self.last_try = self.last_line

    def _drop(self, reason: str):
        if self.ser is not None:
            try: self.ser.close()
            except Exception: pass
        self.ser = None
        self.events.append(dict(t=round(self.clock(), 3), event='drop', reason=reason))
        if reason == 'silent': self.silent_reopens += 1
        else: self.errors += 1

    def _reopen(self):
        now = self.clock()
        if now-self.last_try < RETRY_S: return
        self.last_try = now
        try:
            self.ser = self.opener()
        except Exception as exc:
            self.ser = None; self.events.append(dict(t=round(now, 3), event='reopen_failed', reason=str(exc)[:120])); return
        self.reopens += 1; self.last_line = self.clock()
        self.events.append(dict(t=round(now, 3), event='reopened'))

    def readline(self) -> bytes:
        if not self.is_open: raise OSError(f'{self.port} 已關閉')
        if self.ser is None:
            self._reopen()
            if self.ser is None:
                time.sleep(min(self.timeout, 0.2)); return b''
        try:
            line = self.ser.readline()
        except Exception:
            self._drop('error'); return b''
        now = self.clock()
        if line:
            self.last_line = now; return line
        if now-self.last_line > SILENT_REOPEN_S and now-self.last_try >= RETRY_S:
            self._drop('silent'); self.last_try = 0.0; self._reopen()
        return b''

    def stats(self) -> dict:
        return dict(port=self.port, transport='serial', reopens=self.reopens, read_errors=self.errors,
                    silent_reopens=self.silent_reopens, events=list(self.events)[-50:])

    def close(self):
        self.is_open = False
        if self.ser is not None:
            try: self.ser.close()
            except Exception: pass
            self.ser = None
