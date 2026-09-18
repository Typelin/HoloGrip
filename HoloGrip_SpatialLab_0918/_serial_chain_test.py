import sys,time,serial
from pathlib import Path
root=Path(r'C:\Users\Typelin_Station\Desktop\HoloGrip')
app=root/'Apps'/'Song_Collection_COM'
sys.path.insert(0,str(app))
from product_hit_and_zone import LiveRecognizer,load_zone_model
from song_collection_server import parse_sensor_line
model=root/'Data'/'Derived'/'Song_Collection_COM'/'S20260812_P01_song02_T1127_cleaned'/'hologrip_song2_七鼓點模型_真正驗證版_0826.joblib'
clf=load_zone_model(str(model))
rec=LiveRecognizer(clf,require_calibration=True)
s=serial.Serial('COM4',460800,timeout=0.2)
time.sleep(1.0)
s.reset_input_buffer()
parsed=bad=0
first=[]
t0=time.monotonic()
while time.monotonic()-t0<3.0:
    raw=s.readline()
    if not raw:
        continue
    pkt=parse_sensor_line(raw.decode('utf-8','ignore'),int(time.time()*1000),time.monotonic())
    if pkt is None:
        bad+=1
        continue
    parsed+=1
    if len(first)<3:
        first.append((pkt.hand,pkt.packet_id,pkt.yaw,pkt.pitch))
    rec.push(pkt)
print('PARSED',parsed,'BAD',bad,'FIRST',first)
print('LAST_R',rec.hands['R'].last_packet is not None,'LAST_L',rec.hands['L'].last_packet is not None)
print('CALIBRATE',rec.calibrate())
last=None
t1=time.monotonic()
while time.monotonic()-t1<1.0:
    raw=s.readline()
    if not raw:
        continue
    pkt=parse_sensor_line(raw.decode('utf-8','ignore'),int(time.time()*1000),time.monotonic())
    if pkt:
        rec.push(pkt)
        last=pkt
if last:
    st=rec.hands[last.hand]
    print('AFTER_CAL',last.hand,st.detector.calibrated(last))
s.close()
