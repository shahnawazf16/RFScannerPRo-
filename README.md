RF Scanner Pro 🛰️

https://img.shields.io/badge/RF-Scanner%2520Pro-blue
https://img.shields.io/badge/Python-3.8%252B-green
https://img.shields.io/badge/Open%2520Source-Yes-brightgreen
https://img.shields.io/badge/License-MIT-lightgrey

A comprehensive, open-source RF spectrum analyzer and signal intelligence platform with advanced machine learning capabilities for analyzing various radio frequency signals including ADS-B, GPS, GNSS, GSM, satellite, radar, and more.
🌟 Features
📡 Multi-Band RF Scanning

    ADS-B (1090 MHz) - Aircraft tracking and monitoring

    GPS/GNSS (L1/L2 bands) - Satellite navigation signals

    GSM (900/1800 MHz) - Cellular communications

    Satellite Communications (L-Band)

    Radar Systems (S-Band)

    Custom Frequency Ranges (1-6000 MHz)

🔬 Advanced Signal Analysis

    Real-time Spectrum Analysis with FFT

    Waterfall Display for time-frequency analysis

    Constellation Diagrams for modulation visualization

    Signal Parameter Extraction (SNR, bandwidth, center frequency)

    Modulation Recognition (AM, FM, PSK, QPSK, QAM, FSK, OFDM)

🤖 Machine Learning Integration

    Automatic Modulation Classification

    Signal Type Recognition

    Anomaly Detection

    Pattern Recognition

🎛️ Professional GUI

    Tabbed Interface for multiple visualizations

    Real-time Controls and parameter adjustment

    Data Recording capabilities

    Signal Information Display

🚀 Quick Start
Installation
bash

# Clone the repository
git clone https://github.com/yourusername/rf-scanner-pro.git
cd rf-scanner-pro

# Install required packages
pip install numpy scipy matplotlib PyQt5 pyqtgraph scikit-learn

Run the Application
bash

python3 RFScannerPro.py

📋 Requirements

Create a requirements.txt file in your project directory:
txt

numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
PyQt5>=5.15.0
pyqtgraph>=0.12.0
scikit-learn>=1.0.0

System Requirements

    Python: 3.8 or higher

    RAM: 4GB minimum (8GB recommended)

    Storage: 1GB free space

    OS: Windows, Linux, or macOS

💻 Usage Guide
First Time Setup

    Install Python dependencies as shown above

    Run python3 RFScannerPro.py

    The application will start with simulated RF data

Basic Operation

    Select Frequency Band: Choose from dropdown (ADS-B, GPS, GSM, etc.)

    Adjust Parameters: Set gain, bandwidth, and sample rate

    Start Scanning: Click "Start Scanning" button

    View Results: Switch between different visualization tabs

Visualization Tabs

    Waterfall: Time-frequency analysis display

    Spectrum: Real-time frequency spectrum

    FFT: Fast Fourier Transform analysis

    Constellation: IQ diagram for modulation analysis

    Signal Info: ML classification results and signal parameters

Machine Learning Analysis

Click "ML Analysis" button to:

    Automatically detect modulation type

    Classify signal type (ADS-B, GPS, GSM, etc.)

    Display signal parameters and quality metrics

🛠️ Supported Hardware
SDR Devices (Optional)

For real RF signal reception, you can use:

    RTL-SDR (Recommended for beginners)

    NooElec NESDR Smart

    ADALM-PLUTO

    HackRF One

Simulation Mode

The application includes a sophisticated simulation mode that generates realistic RF signals for:

    ADS-B aircraft signals

    GPS satellite navigation

    GSM cellular communications

    Various modulation types

🔧 Technical Details
Signal Processing

    Sample Rates: 1-20 MHz configurable

    FFT Analysis: 1024-point Welch method

    Frequency Range: 1-6000 MHz simulation

    Real-time Processing: Multi-threaded architecture

Machine Learning Features

    Modulation Classification: Neural Network based

    Signal Type Detection: Random Forest classifier

    Feature Extraction: Spectral and temporal analysis

Supported Signals
Signal Type	Frequency	Modulation
ADS-B	1090 MHz	Pulsed
GPS L1	1575.42 MHz	CDMA
GLONASS	1602 MHz	FDMA
GSM 900	925-960 MHz	GMSK
GSM 1800	1805-1880 MHz	GMSK
🗂️ Project Structure
text

RFScannerPro/
│
├── RFScannerPro.py          # Main application file
├── requirements.txt          # Python dependencies
├── README.md                # This file
│
└── (No additional files needed - single file application!)

🎯 Use Cases
🛩️ Aviation Enthusiasts

    Monitor ADS-B signals from aircraft

    Track flight patterns in real-time

    Analyze aircraft transponder data

📡 Radio Amateurs

    Explore RF spectrum usage

    Analyze signal characteristics

    Learn about different modulation schemes

🔬 Educational Purposes

    Signal processing demonstrations

    Machine learning in RF applications

    Digital communications coursework

🛡️ Security Research

    RF spectrum monitoring

    Signal identification and classification

    Anomaly detection in RF environment

🤝 Contributing

We welcome contributions from the community!
How to Contribute

    Fork the repository

    Create a feature branch

    Make your changes

    Submit a pull request

Areas for Improvement

    Additional SDR hardware support

    More ML models and features

    Enhanced visualization options

    Performance optimizations

❓ Frequently Asked Questions

Q: Do I need an SDR device to use this software?
A: No! The software includes a comprehensive simulation mode that generates realistic RF signals for testing and demonstration.

Q: What Python version is required?
A: Python 3.8 or higher is recommended.

Q: Can I use this for commercial purposes?
A: Yes! This software is completely free and open source.

Q: How accurate is the ML classification?
A: The ML features provide good demonstration accuracy. For production use, additional training with real-world data is recommended.
📜 License

This project is released under the MIT License - feel free to use, modify, and distribute as you see fit.
⚠️ Disclaimer

This software is intended for educational, research, and legitimate spectrum monitoring purposes only. Users are responsible for complying with local regulations regarding RF transmission and reception.

Ready to explore the RF spectrum? Run python3 RFScannerPro.py and start scanning!

For issues and questions, please open an issue on GitHub.
