"""Dedicated launcher for Raw 100 Hz wired HoloGrip song collection.

This entry does not load the seven-drum model and does not run live
classification.  Model validation has its own launcher.
"""

try:
    from .song_collection_ui import DesignedSongCollectionApp
except ImportError:
    from song_collection_ui import DesignedSongCollectionApp


def main() -> None:
    app = DesignedSongCollectionApp(transport="serial")
    # Formal collection always starts in Raw 100 Hz mode.  The operator may
    # still inspect the alternate mode, but a fresh launch cannot silently
    # default to filtered hit events.
    app.mode_var.set("raw_100hz")
    if hasattr(app, "mode_segment"):
        app.mode_segment.set("原始 100 Hz")
    app._update_mode_description()
    app.mainloop()


if __name__ == "__main__":
    main()
