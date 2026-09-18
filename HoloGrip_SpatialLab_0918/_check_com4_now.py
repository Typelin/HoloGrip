import serial
try:
    s=serial.Serial('COM4',460800,timeout=.2)
    print('OPEN_OK')
    s.close()
except Exception as e:
    print('OPEN_FAIL')
    print(type(e).__name__)
    print(repr(e))
    print('winerror=',getattr(e,'winerror',None))
