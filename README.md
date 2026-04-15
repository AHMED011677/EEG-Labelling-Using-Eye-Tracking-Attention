# Gaze-Based EEG Annotation System

This project explores the use of eye-tracking data to automate the annotation of EEG signals. By capturing expert gaze behaviour, the system generates attention-based labels to improve annotation speed and consistency.

## Features
- Fixation detection using I-DT algorithm
- Gaze clustering using DBSCAN and Gaussian Mixture Models (GMM)
- Heatmap generation to visualise attention
- Mapping gaze coordinates to EEG time and channels
- Automated label generation (CSV/JSON)
- EEG windowing and feature extraction
- Classification using Random Forest

## Tech Stack
- Python
- NumPy, SciPy
- scikit-learn
- MNE (EEG processing)
- OpenCV
- matplotlib

## Project Structure
- `src/` – core pipeline (preprocessing, analysis, modelling)
- `results/` – generated figures and tables
- `models/` – trained ML models

## How it works
1. Extract fixations from gaze data
2. Segment into 2-minute crops
3. Cluster gaze points (DBSCAN/GMM)
4. Generate heatmaps and attention regions
5. Map gaze to EEG signals
6. Train classifier on labelled EEG windows

## Example Outputs
(Add your images here – heatmaps, EEG overlays, etc.)

## Future Work
- Compare gaze-based labels with expert hand-labelled EEG data
- Improve temporal alignment between gaze and EEG
