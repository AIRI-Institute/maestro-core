from tabulate import tabulate
from chat_manager_examples.config import TRACKS_MODULE
from mmar_mapi.tracks import load_tracks


def get_path(module):
    return module.__module__.replace(".", "/") + ".py"


def find_and_show_tracks():
    tracks = load_tracks(TRACKS_MODULE)
    tbl = [
        (getattr(track, "CLIENTS", "any"), track.DOMAIN, track.__name__, track.CAPTION, 'src/' + get_path(track))
        for track in tracks.values()
    ]
    tbl.sort(key=lambda x: str(x[1]))
    headers = ("clients", "domain", "track_id", "caption", "path")
    try:
        tbl_pretty = tabulate(tbl, headers=headers)
    except ImportError:
        tbl_pretty = "\n".join([" | ".join(headers), "-----"] + list(" | ".join(map(str, row)) for row in tbl))

    print(tbl_pretty)
    print(f"Total: {len(tbl)} tracks")


if __name__ == "__main__":
    find_and_show_tracks()
