import serial,time,threading
ports=['COM4','COM5']
res={}

def readp(p):
    out={'lines':0,'parsed':0,'first':[],'hands':set(),'ids':[],'err':None}
    try:
        s=serial.Serial(p,460800,timeout=0.15)
        time.sleep(.3)
        s.reset_input_buffer()
        t0=time.monotonic(); last_id=None
        while time.monotonic()-t0<3.0:
            b=s.readline()
            if not b: continue
            out['lines']+=1
            line=b.decode('utf-8','replace').strip()
            if len(out['first'])<5: out['first'].append(line)
            f=line.split(',')
            if len(f)>=10 and f[0]=='D' and f[1] in ('L','R'):
                out['parsed']+=1; out['hands'].add(f[1]);
                try: last_id=int(f[8]); out['ids'].append(last_id)
                except: pass
        out['elapsed']=time.monotonic()-t0
        s.close()
    except Exception as e:
        out['err']=repr(e)
    res[p]=out
ths=[threading.Thread(target=readp,args=(p,)) for p in ports]
for t in ths:t.start()
for t in ths:t.join()
for p in ports:
    o=res[p]
    print('===',p,'===')
    print('ERR',o['err'])
    print('LINES',o['lines'],'PARSED',o['parsed'],'HANDS',sorted(o['hands']))
    if o.get('elapsed'): print('HZ',round(o['parsed']/o['elapsed'],1))
    if len(o['ids'])>=2: print('ID_DELTA',o['ids'][-1]-o['ids'][0],'COUNT',len(o['ids']))
    for x in o['first']: print(x)
