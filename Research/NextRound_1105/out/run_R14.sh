# 11/17 night retrain with R14 (11/17 人員3: song 1 + two single-hand drills). usage: run_R14.sh <tag> [build args]
# Scores replay the 11-17的包 runtime (position-prior floor 0.1), the engine that ships.
cd /c/Users/Typelin_Station/Desktop/HoloGrip/Research/NextRound_1105
export PYTHONIOENCODING=utf-8 HOLOGRIP_ROUNDS=out/rounds_R14.json
PY=/c/Users/Typelin_Station/AppData/Local/Programs/Python/Python312/python.exe
T=$1; shift; X="R9:ton1轉ton2 L,R12:高音轉中音 L,R10:中音轉Ride L${XEXTRA}"; L=out/log_R14_$T.log; : > $L
timeout 2400 $PY build_model.py --tag $T --exclude "$X" "$@" > out/log_build_$T.log 2>&1; echo build $? >> $L
timeout 1800 $PY score.py --code 11-17的包 --model out/model_$T.joblib --tag pkg_$T > out/log_score_pkg_$T.log 2>&1; echo score $? >> $L
echo done >> $L
