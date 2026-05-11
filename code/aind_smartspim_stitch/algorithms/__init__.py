"""
Stitching algorithm implementations for SmartSPIM datasets.

Two algorithms are supported:

- **terastitcher**: CPU-based, MPI-parallel stitching using the TeraStitcher CLI tool.
  Produces XML displacement maps and merged OME-Zarr volumes.

- **bigstitcher**: Java/Spark-based stitching using BigStitcher (via run_classes.sh).
  Computes phase-correlation tile transforms and writes BigDataViewer-compatible XML.
"""
