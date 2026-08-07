# HoloGrip SONG Collection (COM)

This is the dedicated wired collection entry for the music-session workflow.

Launch `run_song_collection_com.bat`.

New raw CSV files are written to `Data/Raw/Song_Collection_COM/`.

The separate `Apps/UDP_Collection/` entry remains for the legacy Wi-Fi/UDP workflow.

## MIDI label pipeline

Run `run_midi_label_pipeline_0805.bat` for the current session. It reads the
Raw CSV and the MIDI source, applies the confirmed 110 BPM override and the
0805 seven-zone mapping, then writes reviewable derived files under
`Data/Derived/Song_Collection_COM/`.

The source CSV and MIDI are never overwritten. The generated labels remain
`pending_manual_alignment_review` until the selected offset is checked against
the recording or a known sync point.
