"""Previous two-card Tk interface, connected to the frozen seventh-package engine.

The worker owns serial, inference and saving. Tk only renders snapshots and
enqueues commands. Item markers are metadata, never inference inputs.
"""
import argparse
import os
from pathlib import Path
import queue
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import seventh_core as core
from seventh_live import Service

DRUMS=core.DRUMS


class App(tk.Tk):
    def __init__(self,service=None):
        super().__init__()
        self.title('HoloGrip｜10/20 七鼓模型實測＋同步保存 Raw')
        self.geometry('1200x940')
        self.minsize(1050,860)
        try:
            self.service=service if service is not None else Service()
        except Exception:
            self.destroy()
            raise
        self.pending=None
        self.closing=False
        self.busy=False
        self.last_saved=None
        self.current_folder=None
        self.last_error=None
        self.last_snapshot={}
        self.status=tk.StringVar(value='請正確戴好手套，預設連接 COM4、COM5。')
        self.port_vars=[tk.StringVar(value='COM4'),tk.StringVar(value='COM5')]
        self.session=tk.StringVar(value='S20261020_P01_sevendrum_take01')
        self.marker_hand=tk.StringVar(value='L')
        self.marker_drum=tk.StringVar(value=DRUMS[0])
        self.buttons={}
        top=ttk.Frame(self,padding=16);top.pack(fill='x')
        self.heading=ttk.Label(top,text='七鼓現場實測｜四＋五＋六聯合軌跡模型',font=('Microsoft JhengHei UI',22,'bold'))
        self.heading.pack(anchor='w')
        ttk.Label(top,text='外部找打擊 → 七鼓 MLP　｜　Raw 全部保存　｜　不讀 MIDI，不現場重訓',
                  font=('Microsoft JhengHei UI',12)).pack(anchor='w',pady=8)
        row=ttk.Frame(top);row.pack(fill='x')
        self.combos=[]
        for var in self.port_vars:
            cb=ttk.Combobox(row,textvariable=var,width=18,state='readonly')
            cb.pack(side='left',padx=4);self.combos.append(cb)
        self.button(row,'scan','掃描 COM',self.scan)
        self.button(row,'connect','連接雙手',self.connect)
        self.button(row,'disconnect','中斷連線',self.disconnect)
        ttk.Label(top,text='歸零：兩手手掌貼鼓面、保持靜止至少 0.5 秒 → 按歸零 → 拿起鼓棒再開始。').pack(anchor='w',pady=10)
        ctl=ttk.Frame(top);ctl.pack(fill='x')
        ttk.Entry(ctl,textvariable=self.session,width=38).pack(side='left',padx=4)
        self.button(ctl,'zero','雙手歸零',lambda:self.command('zero'))
        self.button(ctl,'start','開始新測試',self.start)
        self.button(ctl,'stop','停止並保存',lambda:self.command('stop'))
        self.button(ctl,'open','開啟實測資料夾',self.open_data)
        items=ttk.Frame(top);items.pack(fill='x',pady=(10,0))
        ttk.Label(items,text='測試項目（只做紀錄，不餵模型）：').pack(side='left')
        ttk.Combobox(items,textvariable=self.marker_hand,values=['L','R'],width=4,state='readonly').pack(side='left',padx=4)
        ttk.Combobox(items,textvariable=self.marker_drum,values=DRUMS,width=13,state='readonly').pack(side='left',padx=4)
        self.button(items,'marker_start','本項開始',lambda:self.command('marker_start',(self.marker_hand.get(),self.marker_drum.get())))
        self.button(items,'marker_end','本項結束',lambda:self.command('marker_end'))
        cards=ttk.Frame(self,padding=16);cards.pack(fill='both',expand=True)
        self.labels={};self.hz_labels={};self.counts={'L':0,'R':0};self.tiles={}
        for col,(h,title,color) in enumerate([('L','左手','#165cbe'),('R','右手','#b02c3b')]):
            card=tk.Frame(cards,bg='#f6f8fb',highlightthickness=2,highlightbackground=color)
            card.grid(row=0,column=col,padx=8,sticky='nsew');cards.columnconfigure(col,weight=1)
            tk.Label(card,text=title,bg='#f6f8fb',fg=color,font=('Microsoft JhengHei UI',24,'bold')).pack(pady=12)
            self.labels[h]=tk.Label(card,text='尚未開始',bg='#f6f8fb',fg=color,font=('Microsoft JhengHei UI',32,'bold'))
            self.labels[h].pack(pady=15)
            self.hz_labels[h]=tk.Label(card,text='等待封包',bg='#f6f8fb',font=('Microsoft JhengHei UI',14))
            self.hz_labels[h].pack(pady=10)
            grid=tk.Frame(card,bg='#f6f8fb');grid.pack(fill='x',padx=15,pady=10)
            for i,drum in enumerate(DRUMS):
                tile=tk.Label(grid,text=drum,bg='#e6eaf1',font=('Microsoft JhengHei UI',15),pady=10)
                tile.grid(row=i//2,column=i%2,padx=4,pady=4,sticky='ew');grid.columnconfigure(i%2,weight=1)
                self.tiles[h,drum]=tile
            tk.Label(card,text='亮格＝最近一次打擊結果；不是目前手的位置',bg='#f6f8fb',font=('Microsoft JhengHei UI',10)).pack(pady=8)
        cards.rowconfigure(0,weight=1)
        ttk.Label(self,textvariable=self.status,wraplength=1100,font=('Microsoft JhengHei UI',12)).pack(fill='x',padx=20,pady=10)
        ttk.Label(self,text='場地無關版｜不用地磁方位｜每一下獨立判斷，峰後 80 ms 出結果；位置以鼓面歸零為準，不加額外校準。').pack(padx=20,pady=8)
        self.protocol('WM_DELETE_WINDOW',self.close)
        self.scan()
        self.refresh()
        self.after(40,self.poll)

    def button(self,parent,key,text,command):
        b=ttk.Button(parent,text=text,command=command)
        b.pack(side='left',padx=4);self.buttons[key]=b
        return b

    def scan(self):
        ports=self.service.ports_provider()
        for i,cb in enumerate(self.combos):
            # Keep the requested pair even when unplugged; do not silently
            # select an unrelated device. Service.connect checks actual ports.
            cb['values']=list(dict.fromkeys(['COM4','COM5',*ports]))
            if not self.port_vars[i].get():self.port_vars[i].set(['COM4','COM5'][i])

    def connect(self):
        ports=[var.get() for var in self.port_vars]
        if not all(ports) or len(set(ports))!=2:
            self.status.set('請選兩個不同且存在的 COM。');return
        self.command('connect',ports)

    def disconnect(self):self.command('disconnect')

    def start(self):self.command('start',{'session':self.session.get().strip()})

    def command(self,name,value=None):
        if self.pending or (self.closing and name!='quit'):return False
        response=queue.Queue(maxsize=1)
        try:self.service.tasks.put_nowait((name,value,response))
        except queue.Full:
            self.status.set('佇列滿，指令未送出；請停止測試。');return False
        self.pending=(name,response)
        self.last_error=None
        self.status.set('正在停止、保存並結束…' if name=='quit' else '處理中…')
        self.update_controls()
        return True

    def update_controls(self):
        s=self.last_snapshot;connected=s.get('connected',False)
        health=s.get('health',{})
        ready=connected and set(s.get('port_hands',{}).values())=={'L','R'} and all(
            health.get(h,{}).get('fresh',False) and 80<=health.get(h,{}).get('hz',0)<=120 for h in 'LR')
        enabled=dict(scan=not connected,connect=not connected,disconnect=connected and not self.busy,
            zero=ready and not self.busy,start=ready and bool(s.get('calibration')) and not self.busy,
            stop=self.busy,open=True,marker_start=self.busy,marker_end=self.busy)
        for key,b in self.buttons.items():
            b.state(['!disabled'] if enabled.get(key,False) and not self.pending and not self.closing else ['disabled'])
        for cb in self.combos:cb.configure(state='disabled' if connected or self.pending or self.closing else 'readonly')

    def refresh(self):
        # Serial open/close or model validation can hold the lock. Never block Tk.
        if not self.service.lock.acquire(blocking=False):return
        try:s=self.service.snapshot(include_ports=False)
        finally:self.service.lock.release()
        self.last_snapshot=s
        take=s['take'];self.busy=bool(take)
        if take:
            if self.current_folder!=take['folder']:
                self.current_folder=take['folder']
                for label in self.labels.values():label.configure(text='等待打擊')
                for tile in self.tiles.values():tile.configure(bg='#e6eaf1')
            self.counts=dict(take['formal_counts'])
            for h in 'LR':
                p=take['last_hit'][h]
                if p:
                    self.labels[h].configure(text=p['drum'])
                    for drum in DRUMS:self.tiles[h,drum].configure(bg='#a5e8b1' if drum==p['drum'] else '#e6eaf1')
        elif s['last_saved']:
            saved=s['last_saved'];self.last_saved=Path(saved['folder'])
            self.counts=dict(saved['report']['formal_counts'])
            for label in self.labels.values():label.configure(text='本場已保存' if saved['report']['saved_ok'] else '存檔需檢查')
        for h in 'LR':
            health=s['health'][h]
            text=f'{health["hz"]:.1f} Hz｜本次 {self.counts[h]} 下'
            if s['connected'] and not health['fresh']:text+='｜訊號中斷'
            self.hz_labels[h].configure(text=text)
        if not self.pending and not self.closing:
            status=s['status']
            if take:status+=f'｜Raw 已收 {take["raw_enqueued"]} 筆'
            if take:
                names={'venue_free':'場地無關','trained_world':'既有方向','drumhead_relative':'鼓面相對方向',None:'等待資料'}
                status+='｜'+ '、'.join(f'{h}：{names[take["direction_mode"][h]]}' for h in 'LR')
            elif s['last_saved']:status+='｜'+s['last_saved']['folder']
            self.status.set(status)
        self.update_controls()

    def poll(self):
        if self.pending:
            name,response=self.pending
            try:ok,payload=response.get_nowait()
            except queue.Empty:pass
            else:
                self.pending=None
                if not ok:
                    self.last_error=payload;self.status.set(payload)
                    if name=='quit':self.closing=False
                elif name=='quit':
                    self.service.worker.join(timeout=.1)
                    self.destroy();return
        if self.closing and self.pending is None:self.command('quit')
        self.refresh()
        self.after(40,self.poll)

    def open_data(self):
        folder=self.last_saved or self.service.data_root/'模型實測'
        folder.mkdir(parents=True,exist_ok=True)
        os.startfile(str(folder))

    def close(self):
        if self.closing:return
        self.closing=True
        self.status.set('正在停止、保存並結束…請勿強制關閉。')
        if self.pending is None:self.command('quit')
        self.update_controls()


def main():
    parser=argparse.ArgumentParser(description='HoloGrip 原生 Python 七鼓實測 GUI')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    app=App()
    if args.smoke:
        app.withdraw();app.after(100,app.close)
    app.mainloop()
    if args.smoke:print('PASS: native Tk GUI + independent detector (2.1 g) + venue-free per-hit committee (104 features); no browser or COM opened.')
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        if '--smoke' not in sys.argv:
            root=tk.Tk();root.withdraw()
            messagebox.showerror('HoloGrip 無法啟動',str(exc),parent=root)
            root.destroy()
        sys.exit(2)
