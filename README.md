# Sea-Ice-Multi-Sensor-Pipeline
An automated data pipeline for sea ice thickness retrieval
# 🌊 Sea Ice Thickness Retrieval Pipeline: Multi-Sensor Data Fusion

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-API-green)](#)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange)](#)

## 📖 Overview
This repository contains an end-to-end automated data pipeline for sea ice thickness retrieval, developed as part of the National Key R&D Program of China. It integrates multi-source satellite data, specifically combining Sentinel-1 (S1) SAR imagery with CryoSat-2 (CS2) radar altimetry waveforms. [cite_start]The pipeline handles large-scale data matching [cite: 47][cite_start], cloud-based feature extraction via Google Earth Engine (GEE) [cite: 49][cite_start], and machine learning classification to accurately differentiate sea ice leads and floes[cite: 55, 60].

## ✨ Key Features
* **Automated Data Matching & Extraction:** Batch downloads Sentinel-1 SAFE metadata and processes spatial-temporal overlaps with CryoSat-2 NetCDF files.
* **Cloud-Native Preprocessing:** Utilizes the Google Earth Engine API to export S1 imagery to Google Drive, calculating specific bands and masking by sea ice concentration (SIC > 70%).
* **Machine Learning Classification:** Implements Random Forest (RF) classification for Sentinel-1 imagery and classifies CryoSat-2 L1 waveform points.
* **Robust Labeling Workflow:** Integrates ArcGIS Pro for spatial labeling, utilizing Python (`datetime.datetime.strptime`) to format time profiles properly for GEE compatibility.

## 📂 Repository Structure
```text
├── bulk_download/          # Scripts for Sentinel-1 SAFE metadata downloading
├── CS2_S1_match/           # Spatiotemporal matching of S1 and CS2 footprint pairs
├── S1_S2_overlap/          # Overlap processing and validation scripts
├── classification/         # Random Forest modeling and S1 image classification
├── CS2_L1_classifi/        # CryoSat-2 Level-1 waveform feature extraction and classification
└── data_processing/        # Includes NetCDF to TIFF conversion and offset handling