import argparse
from opensfm import commands
import sys

if len(sys.argv) < 2:
    raise IOError("Usage run_pipeline <data_dir>")
data_dir = sys.argv[1]

args = argparse.Namespace(dataset=data_dir)

# Run through every module
mod = {module.Command().name: module for module in commands.opensfm_commands}
mod['extract_metadata'].Command().run(args=args)
mod['detect_features'].Command().run(args=args)
mod['match_features'].Command().run(args=args)
mod['create_tracks'].Command().run(args=args)
mod['reconstruct'].Command().run(args=args)