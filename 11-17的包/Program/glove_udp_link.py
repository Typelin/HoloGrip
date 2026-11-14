"""Wireless glove input (UDP) that behaves like the two wired COM ports.

Gloves (Firmware/Gloves_*_UDP) send the same text line as the wired firmware,
    D,HAND,ax,ay,az,yaw,pitch,roll,packet_id,sensor_millis
at 100 Hz to UDP port 8888. Each datagram carries the newest line plus the previous
REDUNDANT-1 lines, so a lost datagram is recovered from the next one.

Receiver (this module):
  * discovery: broadcasts HELLO to port 8889 every second; a glove that hears it sends
    unicast to this computer (unicast gets Wi-Fi link-layer retries, broadcast does not).
  * per hand: duplicates dropped by packet_id, short reordering held up to HOLD_MS, a
    packet still missing after that is skipped (the take counts it as a packet gap).
  * timing: received_monotonic is placed on the glove's own clock,
        t = sensor_millis/1000 + offset,  offset = min(arrival - sensor_time) over 5 s,
    so samples keep their real 10 ms spacing even when the network delivers them in bursts;
    it never lies after the actual arrival. A firmware restart (packet_id or clock going
    back) restarts the mapping.
  * UdpPort('UDP-L' / 'UDP-R') looks like a serial port to the existing programs:
    .port, .is_open, .close(), .read_packet(timeout) -> SensorPacket or None,
    .readline() -> the original text line (bytes) for code that only reads lines.
"""
from __future__ import annotations

import queue
import socket
import threading
import time
from collections import deque
from typing import Optional

from song_collection_server import SensorPacket, parse_sensor_line

DATA_PORT = 8888
HELLO_PORT = 8889
HELLO = b'HG_HELLO'
HOLD_MS = 60.0            # wait this long for a missing packet_id before skipping it
OFFSET_WINDOW_S = 5.0     # arrival - sensor time minimum is taken over this window
PORT_NAMES = {'UDP-L': 'L', 'UDP-R': 'R'}


def broadcast_addresses() -> list[str]:
    """255.255.255.255 plus the /24 broadcast of every local IPv4 address (phone hotspots are /24)."""
    out = ['255.255.255.255']
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith('127.'):
                out.append('.'.join(ip.split('.')[:3]+['255']))
    except OSError:
        pass
    return list(dict.fromkeys(out))


class _HandStream:
    """Dedup + reorder + clock mapping for one hand."""

    def __init__(self, hand: str):
        self.hand = hand
        self.out: queue.Queue = queue.Queue(maxsize=20000)
        self.lines: queue.Queue = queue.Queue(maxsize=20000)
        self.received = 0; self.duplicates = 0; self.skipped = 0; self.restarts = 0
        self.last_arrival: Optional[float] = None
        self.reset()

    def reset(self):
        """Forget packet order and clock mapping (start, or the glove restarted); counters are kept."""
        self.next_id: Optional[int] = None
        self.pending: dict[int, tuple] = {}
        self.first_wait: Optional[float] = None
        self.offsets: deque = deque()
        self.last_t = -1e18
        self.last_sensor: Optional[int] = None

    def stats(self) -> dict:
        return dict(received=self.received, duplicates=self.duplicates, skipped=self.skipped, restarts=self.restarts,
                    last_arrival=self.last_arrival)

    def feed(self, line: str, pkt: SensorPacket, arrival: float, wall: float):
        self.received += 1; self.last_arrival = arrival
        pid = pkt.packet_id
        if self.next_id is not None and (pid < self.next_id-200 or (self.last_sensor is not None and pkt.sensor_time_ms < self.last_sensor-1000)):
            self._flush(arrival, force=True); self.reset(); self.restarts += 1
        if self.next_id is None: self.next_id = pid
        if pid < self.next_id or pid in self.pending:
            self.duplicates += 1; return
        self.pending[pid] = (line, pkt, arrival, wall)
        self._flush(arrival)

    def _flush(self, now: float, force: bool = False):
        while self.pending:
            if self.next_id in self.pending:
                line, pkt, arrival, wall = self.pending.pop(self.next_id); self.next_id += 1; self.first_wait = None
                self._emit(line, pkt, arrival, wall)
                continue
            if self.first_wait is None: self.first_wait = now
            if force or (now-self.first_wait)*1000 >= HOLD_MS:
                nxt = min(self.pending); self.skipped += nxt-self.next_id; self.next_id = nxt; self.first_wait = None
                continue
            break

    def tick(self, now: float):
        if self.pending: self._flush(now)

    def _emit(self, line: str, pkt: SensorPacket, arrival: float, wall: float):
        s = pkt.sensor_time_ms/1000.0
        self.offsets.append((arrival, arrival-s))
        while self.offsets and arrival-self.offsets[0][0] > OFFSET_WINDOW_S: self.offsets.popleft()
        off = min(o for _, o in self.offsets)
        t = min(max(s+off, self.last_t), arrival)   # monotone, never after the real arrival
        self.last_t = t; self.last_sensor = pkt.sensor_time_ms
        mapped = SensorPacket(hand=pkt.hand, ax=pkt.ax, ay=pkt.ay, az=pkt.az, yaw=pkt.yaw, pitch=pkt.pitch, roll=pkt.roll,
                              packet_id=pkt.packet_id, sensor_time_ms=pkt.sensor_time_ms,
                              received_time_ms=int(round((wall-(arrival-t))*1000)), received_monotonic=t)
        for q, item in ((self.out, mapped), (self.lines, line)):
            try: q.put_nowait(item)
            except queue.Full: pass


class UdpGloveLink:
    """One UDP socket shared by both hands (started on first use)."""

    _instance: Optional['UdpGloveLink'] = None
    _lock = threading.Lock()

    @classmethod
    def shared(cls, **kw) -> 'UdpGloveLink':
        with cls._lock:
            if cls._instance is None or cls._instance.closed: cls._instance = cls(**kw)
            return cls._instance

    def __init__(self, data_port: int = DATA_PORT, hello_port: int = HELLO_PORT, clock=time.monotonic):
        self.clock = clock; self.hello_port = hello_port; self.closed = False
        self.streams = {h: _HandStream(h) for h in 'LR'}
        self.peers: dict[str, str] = {}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.sock.bind(('0.0.0.0', data_port)); self.sock.settimeout(0.02)
        self.stop = threading.Event(); self.users = 0
        threading.Thread(target=self._rx, daemon=True, name='udp-glove-rx').start()
        threading.Thread(target=self._hello, daemon=True, name='udp-glove-hello').start()

    def _rx(self):
        while not self.stop.is_set():
            try:
                data, addr = self.sock.recvfrom(4096)
            except socket.timeout:
                now = self.clock()
                for s in self.streams.values(): s.tick(now)
                continue
            except OSError:
                break
            arrival = self.clock(); wall = time.time()
            for line in data.decode('ascii', errors='replace').splitlines():
                pkt = parse_sensor_line(line, int(wall*1000), arrival)
                if pkt is None or pkt.packet_id < 0: continue
                self.peers[pkt.hand] = addr[0]
                self.streams[pkt.hand].feed(line, pkt, arrival, wall)
            for s in self.streams.values(): s.tick(arrival)   # a stalled hand still releases held packets

    def _hello(self):
        while not self.stop.is_set():
            for addr in broadcast_addresses()+list(self.peers.values()):
                try: self.sock.sendto(HELLO, (addr, self.hello_port))
                except OSError: pass
            self.stop.wait(1.0)

    def port(self, name: str) -> 'UdpPort':
        if name not in PORT_NAMES: raise ValueError(f'未知的無線埠 {name}')
        h = PORT_NAMES[name]; s = self.streams[h]
        for q in (s.out, s.lines):
            while True:
                try: q.get_nowait()
                except queue.Empty: break
        self.users += 1
        return UdpPort(self, name, h)

    def release(self):
        self.users -= 1
        if self.users <= 0: self.close()

    def close(self):
        if self.closed: return
        self.closed = True; self.stop.set()
        try: self.sock.close()
        except OSError: pass


class UdpPort:
    """Serial-port look-alike for one hand."""

    def __init__(self, link: UdpGloveLink, name: str, hand: str):
        self.link = link; self.port = name; self.hand = hand; self.is_open = True

    def read_packet(self, timeout: float = 0.2) -> Optional[SensorPacket]:
        if not self.is_open: raise OSError('無線埠已關閉')
        try: return self.link.streams[self.hand].out.get(timeout=timeout)
        except queue.Empty: return None

    def readline(self) -> bytes:
        if not self.is_open: raise OSError('無線埠已關閉')
        try: return (self.link.streams[self.hand].lines.get(timeout=0.2)+'\n').encode('ascii')
        except queue.Empty: return b''

    def stats(self) -> dict:
        return dict(port=self.port, peer=self.link.peers.get(self.hand), **self.link.streams[self.hand].stats())

    def close(self):
        if self.is_open:
            self.is_open = False; self.link.release()


def open_port(name: str) -> UdpPort:
    return UdpGloveLink.shared().port(name)
