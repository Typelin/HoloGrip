"""Launch the HoloGrip wired USB serial/COM song collector."""

try:
    from .song_collection_ui import DesignedSongCollectionApp
except ImportError:
    from song_collection_ui import DesignedSongCollectionApp


if __name__ == "__main__":
    DesignedSongCollectionApp(transport="serial").mainloop()
