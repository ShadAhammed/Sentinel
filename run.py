"""
run.py

Entry point for the SENTINEL-X demo window.

Run from the project root:

    python run.py

Requires the Jetson Orin NX GPU, the three YOLO weights in
Data/YOLO-Models-Final/, EffNet_b0.pt in Data/, and
Data/video/combined_drone.mp4. See README.md for details.
"""
from sentinel.window import main

if __name__ == "__main__":
    main()
