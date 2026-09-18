import serial,time
s=serial.Serial('COM4',460800,timeout=0.15)
time.sleep(.3)
s.reset_input_buffer()
lines=[]
t=time.time()+1.2
while time.time()<t:
 b=s.readline()
 if b:
  lines.append(b.decode('utf-8','replace').strip())
s.close()
print('LINES',len(lines))
for x in lines[:8]: print(x)
