import serial,time,threading
ports=['COM4','COM5'];res={}
def run(p):
 o={'err':None,'parsed':0,'hands':set(),'stats':[]}
 try:
  s=serial.Serial(p,460800,timeout=.08); time.sleep(1.2); s.reset_input_buffer(); t0=time.monotonic()
  while time.monotonic()-t0<5.0:
   b=s.readline()
   if not b: continue
   line=b.decode('utf-8','replace').strip(); f=line.split(',')
   if line.startswith('#STAT'): o['stats'].append(line)
   if len(f)>=10 and f[0]=='D' and f[1] in ('L','R'):
    o['parsed']+=1;o['hands'].add(f[1])
  o['elapsed']=time.monotonic()-t0;s.close()
 except Exception as e:o['err']=repr(e)
 res[p]=o
ths=[threading.Thread(target=run,args=(p,)) for p in ports]
[t.start() for t in ths];[t.join() for t in ths]
for p in ports:
 o=res[p];print('===',p,'===');print('ERR',o['err']);print('PARSED',o['parsed'],'HANDS',sorted(o['hands']));
 if o.get('elapsed'): print('HZ',round(o['parsed']/o['elapsed'],2));
 [print(s) for s in o['stats'][-3:]]
