"""Dedicated launcher for the wired HoloGrip song collector."""

try:
    from .song_collection_ui import DesignedSongCollectionApp
except ImportError:
    from song_collection_ui import DesignedSongCollectionApp


if __name__ == "__main__":
    DesignedSongCollectionApp(transport="serial").mainloop()
