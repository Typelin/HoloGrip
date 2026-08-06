# MIDI ↔ CSV visualization keepalive

- Phase: Phase 7 delivery preparation
- Result: The formal MIDI-to-glove CSV alignment tool now includes the embedded 0805 review dataset, a local-file loader for Standard MIDI Format 0/1 plus Raw 100Hz CSV, a workflow-first layout, explicit candidate-offset explanation, first/last MIDI event navigation, and CSV/MIDI timeline boundary notes. The loader null-selector error was fixed by anchoring updates to the data section instead of a fragile sibling selector.
- Absolute path: C:\Users\Typelin_Station\Desktop\HoloGrip\Song_Collection_COM\midi_csv_alignment_tool_0805.html
- Algorithm record: C:\Users\Typelin_Station\Desktop\HoloGrip\Song_Collection_COM\ALIGNMENT_ALGORITHM_RECORD_0805.md
- Evidence: JavaScript syntax passed for both script blocks. The real `Drum Midi_110BPM (0805).mid` parsed to 1,820 mapped events at 110 BPM (Format 0, PPQN 96); the real Raw CSV parsed to 74,592 samples and 3,732 display energy points. Simulated loader execution preserved 373,125 ms duration, aligned the first MIDI event to 16,000 ms, completed without the prior null assignment, and produced the first/last navigation controls. The source report records the auto-candidate score 4.4620746 at 16,000 ms.
- Residuals: Browser policy blocked direct `file://` visual screenshot testing, so no browser screenshot was produced. The +16,000 ms offset remains a candidate until verified by a synchronisation strike or WAV. MIDI has no native left/right hand field; hand labels remain CSV-energy candidates.
