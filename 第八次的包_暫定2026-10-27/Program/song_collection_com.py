"""Dedicated launcher for Raw 100 Hz wired HoloGrip song collection.

This entry does not load the seven-drum model and does not run live
classification.  Model validation has its own launcher.
"""

try:
    from .song_collection_ui import DesignedSongCollectionApp
except ImportError:
    from song_collection_ui import DesignedSongCollectionApp


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Build UI without COM or recording')
    args = parser.parse_args()
    app = DesignedSongCollectionApp(transport="serial")
    app.raw_only = True
    # This dedicated field launcher is Raw 100 Hz only; prevent accidental
    # switching to filtered hit events during formal collection.
    app.mode_var.set("raw_100hz")
    if hasattr(app, "mode_segment"):
        app.mode_segment.set("原始 100 Hz")
        app.mode_segment.configure(state="disabled")
    app._update_mode_description()
    app.file_var.set(f"CSV 將儲存到：{app.output_dir.resolve()}")
    if args.smoke:
        app.withdraw()
        app.update_idletasks()
        app.after(150, app._on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
