# KEEPALIVE: dense MIDI event review

## Current result

The 0805 MIDI/CSV alignment page now shows dense MIDI onsets as grouped, overlapping, or single-label events. The Python pipeline emits the same density fields and uses a fixed 90 ms before/after window by default.

## Evidence

- HTML scripts: all 3 script blocks pass JavaScript syntax compilation.
- Python AST parse and self-test pass.
- Temporary real-data pipeline run: 1,820 candidate rows, 398 single-label eligible, 1,422 review events.
- Embedded 3:31.176 data: the 211176.136 ms group is 38 小鼓 + 51 Ride + 46 Hi-Hat, classified as multi_label_overlap at ±90 ms.

## Guardrails

- Do not force dense or simultaneous MIDI events into a single-label training sample.
- Preserve the raw MIDI and Raw CSV files.
- Keep the current 100 Hz data path; 200 Hz is a later data-collection option, not the immediate fix.
- Browser visual automation was unavailable in this session; static and temporary real-data checks were completed.
