# DIVP Queue Monitor

## Overview

DIVP Queue Monitor is a computer vision-based queue monitoring system that uses YOLO and OpenCV to detect and analyze people in a queue from video footage.

The system processes video frames to estimate queue size, monitor waiting time, and generate useful queue-related statistics.

## Features

- Person detection using YOLO
- Real-time/video-based queue monitoring
- Queue size estimation
- Average waiting time calculation
- Video frame processing using OpenCV
- CSV-based result generation
- Automated analysis of queue behavior

## Technologies Used

- Python
- YOLO
- Ultralytics
- OpenCV
- NumPy
- Pandas

## Project Structure

```text
divp-queue-monitor/
│
├── input/
├── output/
├── main.py
├── avg_waiting_time.py
├── requirements.txt
├── README.md
└── .gitignore
