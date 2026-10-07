"""
Build the C++ simulator into src/cpp/_pitwall_sim.*.so.

    .venv/bin/python -m src.cpp.build

Re-run after every change to simulator.cpp (CMake only recompiles what changed).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pybind11

SOURCE_DIR = Path(__file__).resolve().parent
BUILD_DIR = SOURCE_DIR.parents[1] / "build" / "cpp"


def main() -> None:
    subprocess.run(
        [
            "cmake", "-S", str(SOURCE_DIR), "-B", str(BUILD_DIR),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            f"-DPython_EXECUTABLE={sys.executable}",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "-j"], check=True)
    built = sorted(SOURCE_DIR.glob("_pitwall_sim*"))
    print("built:", *built, sep="\n  ")


if __name__ == "__main__":
    main()
