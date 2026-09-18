from pathlib import Path
import csv,sys,warnings,json
from collections import Counter
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
import joblib
warnings.filterwarnings("ignore")

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,ZONE_NAMES
from product_hit_and_zone import ZONE_MAP

RAW12=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT12=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
RAW5=ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"
GT5=ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv"

HMAP={"Crash":0,"Hi-Hat":0,"小鼓":1,"高音 Tom":1,"中音 Tom":1,"Ride":2,"落地 Tom":2}
VMAP={"Crash":0,"高音 Tom":0,"中音 Tom":0,"Ride":0,"Hi-Hat":1,"小鼓":1,"落地 Tom":1}
HN=["LEFT","CENTER","RIGHT"];VN=["UP","LOW"]

def rcsv(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
def prep(rawp):
    rows=rcsv(rawp);out={}
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h]
        t0=min(float(r["song_time_ms"]) for r in rr)
        cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
        R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
        out[h]=[make_relative_sample(R0,float(r["song_time_ms"]),float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
                 float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
    return out
D12=prep(RAW12);D5=prep(RAW5)
def build12():
    X=[];d=[];h=[];g=[]
    for r in rcsv(GT12,"utf-8"):
        hh=r["final_hand"];dr=r["drum"]
        if hh not in ("L","R") or dr not in ZONE_MAP:continue
        c=float(r["csv_center_ms"]);ft=feature_from_samples(D12[hh],c,80)
        if ft is not None:X.append(ft);d.append(dr);h.append(hh);g.append(int(c//4000))
    return np.asarray(X),np.asarray(d),np.asarray(h),np.asarray(g)
def build5():
    X=[];d=[];h=[]
    for r in rcsv(GT5):
        hh=r["assigned_hand"];dr=r["zone_name"]
        if hh not in ("L","R") or dr not in ZONE_MAP:continue
        ft=feature_from_samples(D5[hh],float(r["aligned_csv_time_ms"]),80)
        if ft is not None:X.append(ft);d.append(dr);h.append(hh)
    return np.asarray(X),np.asarray(d),np.asarray(h)
X,d,h,g=build12();X5,d5,h5=build5()

def make(kind):
    if kind=="mlp":return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(32,),alpha=.003,max_iter=1500,random_state=42))])
    if kind=="log":return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,max_iter=3000,class_weight="balanced",random_state=42))])
    if kind=="linear":return Pipeline([("s",StandardScaler()),("m",SVC(C=1,kernel="linear",probability=True,class_weight="balanced",random_state=42))])
    if kind=="rbf":return Pipeline([("s",StandardScaler()),("m",SVC(C=2,gamma="scale",probability=True,class_weight="balanced",random_state=42))])
    if kind=="lda":return Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])

def choose(task,hand):
    idx=np.where(h==hand)[0]
    target=np.array([HMAP[x] for x in d[idx]]) if task=="H" else np.array([VMAP[x] for x in d[idx]])
    best=None
    for kind in ("mlp","log","linear","rbf","lda"):
        pred=np.full(len(idx),-1,int)
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
        try:
            for tr,te in cv.split(X[idx],target,g[idx]):
                m=make(kind);m.fit(X[idx][tr],target[tr]);pred[te]=m.predict(X[idx][te])
        except Exception:continue
        acc=accuracy_score(target,pred);bal=balanced_accuracy_score(target,pred);f1=f1_score(target,pred,average="macro")
        score=bal+.25*acc
        print(task,hand,kind,"CV",round(acc,3),round(bal,3),round(f1,3))
        if best is None or score>best[0]:best=(score,kind,acc,bal,f1)
    return best

out=LAB/"hierarchical_realdrum_models";out.mkdir(parents=True,exist_ok=True)
meta={"models":{}}
models={}
for task in ("H","V"):
  models[task]={}
  meta["models"][task]={}
  for hand in ("L","R"):
    best=choose(task,hand);kind=best[1]
    idx=np.where(h==hand)[0]
    target=np.array([HMAP[x] for x in d[idx]]) if task=="H" else np.array([VMAP[x] for x in d[idx]])
    m=make(kind);m.fit(X[idx],target);models[task][hand]=m
    joblib.dump(m,out/f"{task}_{hand}.joblib")
    meta["models"][task][hand]={"kind":kind,"cv_acc":best[2],"cv_bal":best[3],"cv_f1":best[4]}

print("\n=== CROSS 8/5 COARSE ===")
hp=np.full(len(d5),-1,int);vp=np.full(len(d5),-1,int)
for hand in ("L","R"):
    ii=np.where(h5==hand)[0]
    hp[ii]=models["H"][hand].predict(X5[ii])
    vp[ii]=models["V"][hand].predict(X5[ii])
ht=np.array([HMAP[x] for x in d5]);vt=np.array([VMAP[x] for x in d5])
print("H",accuracy_score(ht,hp),balanced_accuracy_score(ht,hp),f1_score(ht,hp,average="macro"))
print(confusion_matrix(ht,hp))
print("V",accuracy_score(vt,vp),balanced_accuracy_score(vt,vp),f1_score(vt,vp,average="macro"))
print(confusion_matrix(vt,vp))

# 7-drum base MLP
base={hh:joblib.load(LAB/"vnext_models"/f"hologrip_vnext_{hh}_relative_rotation.joblib") for hh in ("L","R")}
raw=np.full(len(d5),-1,int);gated=np.full(len(d5),-1,int)
for i in range(len(d5)):
    hh=h5[i];m=base[hh];pr=m.predict_proba([X5[i]])[0];classes=[int(x) for x in m.classes_]
    raw[i]=classes[int(np.argmax(pr))]
    allowed=[j for j,c in enumerate(classes) if HMAP[ZONE_NAMES[c]]==hp[i] and VMAP[ZONE_NAMES[c]]==vp[i]]
    if allowed:
        j=max(allowed,key=lambda j:pr[j]);gated[i]=classes[j]
    else:gated[i]=raw[i]
y5=np.array([ZONE_MAP[x] for x in d5])
print("\nRAW7",accuracy_score(y5,raw),balanced_accuracy_score(y5,raw),f1_score(y5,raw,average="macro"))
print("GATED7",accuracy_score(y5,gated),balanced_accuracy_score(y5,gated),f1_score(y5,gated,average="macro"))
print("RAW CM\n",confusion_matrix(y5,raw,labels=range(7)))
print("GATED CM\n",confusion_matrix(y5,gated,labels=range(7)))
# physical opposite side
def opp(pred):
    st=[HMAP[x] for x in d5];ps=[HMAP[ZONE_NAMES[int(x)]] for x in pred]
    return sum((a==0 and b==2) or (a==2 and b==0) for a,b in zip(st,ps)),sum(a in(0,2) for a in st)
print("RAW opposite",opp(raw),"GATED opposite",opp(gated))
for k,nm in enumerate(ZONE_NAMES):
    ix=np.where(y5==k)[0]
    if len(ix):
        print(nm,len(ix),"raw",int(np.sum(raw[ix]==k)),"gated",int(np.sum(gated[ix]==k)),
              "gated_preds",Counter(ZONE_NAMES[int(x)] for x in gated[ix]))
(out/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
