import socket
import time
import pycom

from network import Sigfox

sigfox = Sigfox(mode=Sigfox.SIGFOX, rcz=Sigfox.RCZ4)

s = socket.socket(socket.AF_SIGFOX, socket.SOCK_RAW)
s.setblocking(True)
s.setsockopt(socket.SOL_SIGFOX, socket.SO_RX, False)

def zfill(string: str, length: int) -> str:
    """Adds zeroes at the begginning of a string 
    until it completes the desired length."""
    return '0' * (length - len(string)) + string

i = 0
while i < 20000:
    msg = zfill(str(i), 12)
    s.send(bytes(msg, 'utf-8'))
    print('Message sent: {}'.format(msg))
    i = i + 1
    time.sleep(20)

pycom.heartbeat(False)
pycom.rgbled(0x00FF00)