"""Test package for the arc/line vectorization pipeline.

This package is intentionally an empty namespace so the test modules
underneath it are importable. The unit tests live in
``tests/vectorize/low_geometry/tests.py`` and are run by ``test.sh``.

(An older version of this file ran a whole visualization pipeline at
import time — a duplicate of ``test.py`` — which broke whenever
``default_pipeline``'s return signature changed. End-to-end example
processing now lives solely in ``test.py``.)
"""
