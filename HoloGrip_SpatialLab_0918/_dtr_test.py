import serial,time
modes=[(False,False),(True,False),(False,True),(True,True)]
for dtr,rts in modes:
    try:
        s=serial.Serial()
        s.port='COM4'; s.baudrate=460800; s.timeout=0.15
        s.dtr=dtr; s.rts=rts
        s.open()
        time.sleep(1.5)
        n=0; first=None; t=time.time()+1.5
        while time.time()<t:
            b=s.readline()
            if b:
                n+=1
                if first is None: first=b[:120]
        s.close()
        print('MODE',dtr,rts,'LINES',n,'FIRST',first)
        time.sleep(.5)
    except Exception as e:
        print('MODE',dtr,rts,'ERR',repr(e))
