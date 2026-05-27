#!/usr/bin/env bash
# Pipeline test runner.
#   1. Unit tests   — fast geometry/simulator sanity checks. Fail fast:
#                     if these break, the example run is not worth it.
#   2. Example run  — processes every PNG in examples/ end to end.
set -e

echo "=== unit tests ==="
python -m tests.vectorize.low_geometry.tests

echo
echo "=== example pipeline ==="
python -m test
