import serial,time,threading
ports=['COM4','COM5']
res={}
def readp(p):
 o={'err':None,'parsed':0,'hands':set(),'first':[]}
 try:
  s=serial.Serial(p,460800,timeout=.12); time.sleep(.25); s.reset_input_buffer(); t=time.monotonic()
  while time.monotonic()-t<2.5:
   b=s.readline()
   if not b: continue
   line=b.decode('utf-8','replace').strip(); f=line.split(',')
   if len(o['first'])<3:o['first'].append(line)
   if len(f)>=10 and f[0]=='D' and f[1] in ('L','R'):
    o['parsed']+=1;o['hands'].add(f[1])
  o['elapsed']=time.monotonic()-t;s.close()
 except Exception as e:o['err']=repr(e)
 res[p]=o
ths=[threading.Thread(target=readp,args=(p,)) for p in ports]
[t.start() for t in ths];[t.join() for t in ths]
for p in ports:
 o=res[p];print('===',p,'===');print('ERR',o['err']);print('PARSED',o['parsed'],'HANDS',sorted(o['hands']));
 if o.get('elapsed'): print('HZ',round(o['parsed']/o['elapsed'],1))
 [print(x) for x in o['first']]
