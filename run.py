#!/usr/bin/env python
import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Unified entry point: run different third-party model demos by name"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to run (e.g. WHAM, TokenHMR, HMR2.0, etc.)"
    )
    parser.add_argument(
        "--video_pth",
        type=str,
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--output_pth",
        type=str,
        default=None,
        help="Output directory for results (passed along to demo.py if supported)"
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass through to the child demo script"
    )

    args = parser.parse_args()

    # Change working directory to repository root
    repo_root = os.path.dirname(os.path.realpath(__file__))
    os.chdir(repo_root)

    model = args.model.lower()
    video = args.video_pth
    extra = args.extra_args or []

    if model == "wham":
        wham_pth = os.path.join(repo_root, "scripts", "run_WHAM.py")
        cmd = [
            sys.executable, 
            wham_pth,
            "--video", video,
            "--output_pth", args.output_pth
        ]
    
    # Add more elif blocks here for other models like BEDLAM-CLIFF, CameraHMR, NLF, etc.
    else:
        print(f"[run.py] Error: Unknown model name '{args.model}'.")
        sys.exit(1)


    print("Running command:", " ".join(cmd))
    # result = subprocess.run(cmd, cwd=wham_dir)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[run.py] Child process exited with code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()