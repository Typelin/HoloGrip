from __future__ import annotations
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog,messagebox,ttk
import analyze_validation_0917 as ana
ROOT=Path(__file__).resolve().parent.parent
VAL=ROOT/'Data'/'Validation'; MIDI=ROOT/'Data'/'Validation_MIDI'
def newest_session():
    xs=[p for p in VAL.iterdir() if p.is_dir() and (p/'raw_100hz.csv').is_file()] if VAL.exists() else []
    return max(xs,key=lambda p:p.stat().st_mtime) if xs else None
def newest_midi():
    xs=[p for p in MIDI.iterdir() if p.is_file() and p.suffix.lower() in {'.mid','.midi'}] if MIDI.exists() else []
    return max(xs,key=lambda p:p.stat().st_mtime) if xs else None
class App(tk.Tk):
    def __init__(self):
        super().__init__();self.title('HoloGrip 上次模型真實成績 · Raw-only → 最後 MIDI 對答案');self.geometry('980x600');self.minsize(860,520)
        self.session=tk.StringVar(value=str(newest_session() or ''));self.midi=tk.StringVar(value=str(newest_midi() or ''));self.bpm=tk.StringVar();self.tol=tk.StringVar(value='90');self.status=tk.StringVar(value='正式規則：先由 Raw CSV 完成全部 prediction，再讀 MIDI 評分。');self.build()
    def build(self):
        ttk.Label(self,text='② 計算上次模型的真實成績',font=('Microsoft JhengHei UI',22,'bold')).pack(anchor='w',padx=16,pady=(16,4));ttk.Label(self,text='Raw 先獨立完成：有沒有打 + 打哪一鼓。MIDI 最後才揭曉答案，不參與推論。',font=('Microsoft JhengHei UI',13)).pack(anchor='w',padx=16,pady=(0,12))
        f=ttk.Frame(self);f.pack(fill='x',padx=8);f.columnconfigure(1,weight=1)
        ttk.Label(f,text='01 測上次模型產生的資料夾',font=('Microsoft JhengHei UI',12,'bold')).grid(row=0,column=0,sticky='w',padx=12,pady=6);ttk.Entry(f,textvariable=self.session).grid(row=0,column=1,sticky='ew',padx=12,pady=6);ttk.Button(f,text='選資料夾',command=self.pick_session).grid(row=0,column=2,padx=12,pady=6)
        ttk.Label(f,text='同一次模型測試的原始 MIDI',font=('Microsoft JhengHei UI',12,'bold')).grid(row=1,column=0,sticky='w',padx=12,pady=6);ttk.Entry(f,textvariable=self.midi).grid(row=1,column=1,sticky='ew',padx=12,pady=6);ttk.Button(f,text='選 MIDI',command=self.pick_midi).grid(row=1,column=2,padx=12,pady=6)
        ttk.Label(f,text='BPM override（留空=用 MIDI tempo map）').grid(row=2,column=0,sticky='w',padx=12,pady=6);ttk.Entry(f,textvariable=self.bpm,width=12).grid(row=2,column=1,sticky='w',padx=12,pady=6)
        ttk.Label(f,text='配對容差 ms').grid(row=3,column=0,sticky='w',padx=12,pady=6);ttk.Entry(f,textvariable=self.tol,width=12).grid(row=3,column=1,sticky='w',padx=12,pady=6)
        b=ttk.Frame(self);b.pack(fill='x',padx=16,pady=8);ttk.Button(b,text='算上次模型成績：Raw-only → 最後 MIDI 對答案',command=lambda:self.run(True)).pack(side='left',padx=(0,8));ttk.Button(b,text='只重播 Raw，不讀 MIDI',command=lambda:self.run(False)).pack(side='left')
        ttk.Label(self,textvariable=self.status,wraplength=850).pack(fill='x',padx=16,pady=6);self.log=tk.Text(self,height=13,font=('Consolas',11));self.log.pack(fill='both',expand=True,padx=16,pady=(0,16))
    def pick_session(self):
        x=filedialog.askdirectory(initialdir=str(VAL));self.session.set(x or self.session.get())
    def pick_midi(self):
        x=filedialog.askopenfilename(initialdir=str(MIDI),filetypes=[('MIDI','*.mid *.midi'),('All','*.*')]);self.midi.set(x or self.midi.get())
    def run(self,with_midi):
        s=Path(self.session.get().strip())
        if not (s/'raw_100hz.csv').is_file():messagebox.showerror('場次不正確','選擇的資料夾找不到 raw_100hz.csv');return
        m=None
        if with_midi:
            m=Path(self.midi.get().strip())
            if not m.is_file():messagebox.showerror('缺少 MIDI','請選擇同一次演奏的原始 MIDI');return
        try:bpm=float(self.bpm.get()) if self.bpm.get().strip() else None;tol=float(self.tol.get() or 90)
        except ValueError:messagebox.showerror('數值錯誤','BPM / 容差不是有效數字');return
        self.status.set('執行中：先重播 Raw；MIDI 尚不參與推論。')
        def worker():
            try:r=ana.analyze(s,m,bpm=bpm,tol_ms=tol);self.after(0,lambda:self.done(r))
            except Exception as e:self.after(0,lambda:messagebox.showerror('分析失敗',str(e)));self.after(0,lambda:self.status.set('分析失敗；原始資料沒有被覆寫。'))
        threading.Thread(target=worker,daemon=True).start()
    def done(self,r):
        self.log.delete('1.0','end');m=r['inference_manifest'];e=r['midi_evaluation'];self.log.insert('end',f"Raw-only predictions: {m['prediction_count']}\nPrediction SHA-256: {m['prediction_sha256']}\n")
        if e.get('status')=='OK':
            self.log.insert('end',f"MIDI events: {e['midi_events']}\nHit Precision: {e['hit_precision']*100:.1f}%\nHit Recall: {e['hit_recall']*100:.1f}%\nZone accuracy given hit: {e['zone_accuracy_given_hit']*100:.1f}%\nSTRICT Precision: {e['strict_precision']*100:.1f}%\nSTRICT Recall: {e['strict_recall']*100:.1f}%\nSTRICT F1: {e['strict_f1']*100:.1f}%\n");self.status.set('完成：official_validation_report.md 已寫入場次資料夾。')
        else:self.status.set('Raw-only prediction 已固定完成；尚未讀 MIDI。')
        self.log.insert('end',str(Path(r['session'])/ana.REPORT_MD)+'\n');self.log.see('end')
if __name__=='__main__':App().mainloop()
