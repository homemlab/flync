#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import subprocess

# Get the tracks download status
status = os.environ.get('TRACKS_DOWNLOAD_STATUS')

if status != 1:
    raise Exception("❌ Tracks are not downloaded... exiting")

# Open the tracksFile.tsv to read
tracks_file = sys.argv[1]

# Make sure the tracksFile.tsv is sorted
subprocess.run(["sort", "-t$'\t'", "-k3", "-n", tracks_file, "-o", tracks_file])

with open(tracks_file, "r") as f:
    lines = f.readlines()

# Get path for local tracks
appdir = Path(__file__).resolve().parent.parent
local_tracks = Path(appdir) / "static" / "tracks"

# Get the tracks from the local directory
local_tracks_list = [
    f for f in os.listdir(local_tracks) if os.path.isfile(os.path.join(local_tracks, f))
    ]
