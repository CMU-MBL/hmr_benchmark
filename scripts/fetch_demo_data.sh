#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# ( cd third-party/WHAM && bash fetch_demo_data.sh )
( cd third-party/TokenHMR && bash fetch_demo_data.sh )