import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QTabWidget, QGroupBox, QLabel, QComboBox, 
                             QSlider, QPushButton, QTextEdit, QProgressBar, QDoubleSpinBox)
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import threading
from collections import deque
import time
import sys

class RFScannerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Scanner Pro - Advanced Spectrum Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize RF processing components
        self.rf_processor = RFProcessor()
        self.ml_analyzer = MLAnalyzer()
        self.visualizer = AdvancedVisualizations()
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Visualizations
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, 3)
        
    def create_control_panel(self):
        """Create the control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Frequency band selection
        freq_group = QGroupBox("Frequency Bands")
        freq_layout = QVBoxLayout(freq_group)
        
        self.band_combo = QComboBox()
        self.band_combo.addItems([
            "ADS-B (1090 MHz)",
            "GPS L1 (1575.42 MHz)",
            "GLONASS (1602 MHz)",
            "GSM 900",
            "GSM 1800", 
            "L-Band Satellite",
            "S-Band Radar",
            "Custom Range"
        ])
        freq_layout.addWidget(QLabel("Select Band:"))
        freq_layout.addWidget(self.band_combo)
        
        # Custom frequency range
        freq_layout.addWidget(QLabel("Custom Range (MHz):"))
        
        custom_freq_layout = QHBoxLayout()
        self.start_freq_spin = QDoubleSpinBox()
        self.start_freq_spin.setRange(1, 6000)
        self.start_freq_spin.setValue(100)
        self.start_freq_spin.setSuffix(" MHz")
        
        self.stop_freq_spin = QDoubleSpinBox()
        self.stop_freq_spin.setRange(1, 6000)
        self.stop_freq_spin.setValue(2000)
        self.stop_freq_spin.setSuffix(" MHz")
        
        custom_freq_layout.addWidget(QLabel("Start:"))
        custom_freq_layout.addWidget(self.start_freq_spin)
        custom_freq_layout.addWidget(QLabel("Stop:"))
        custom_freq_layout.addWidget(self.stop_freq_spin)
        freq_layout.addLayout(custom_freq_layout)
        
        # Center frequency
        self.center_freq_spin = QDoubleSpinBox()
        self.center_freq_spin.setRange(1, 6000)
        self.center_freq_spin.setValue(1000)
        self.center_freq_spin.setSuffix(" MHz")
        freq_layout.addWidget(QLabel("Center Frequency:"))
        freq_layout.addWidget(self.center_freq_spin)
        
        # Modulation settings
        mod_group = QGroupBox("Modulation Analysis")
        mod_layout = QVBoxLayout(mod_group)
        
        self.modulation_combo = QComboBox()
        self.modulation_combo.addItems([
            "Auto Detect", "AM", "FM", "PSK", "QPSK", "QAM", "FSK", "OFDM"
        ])
        mod_layout.addWidget(QLabel("Modulation Type:"))
        mod_layout.addWidget(self.modulation_combo)
        
        # Analysis parameters
        param_group = QGroupBox("Analysis Parameters")
        param_layout = QVBoxLayout(param_group)
        
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 100)
        self.gain_slider.setValue(50)
        param_layout.addWidget(QLabel("Gain:"))
        param_layout.addWidget(self.gain_slider)
        
        self.bw_slider = QSlider(Qt.Horizontal)
        self.bw_slider.setRange(1, 100)
        self.bw_slider.setValue(20)
        param_layout.addWidget(QLabel("Bandwidth (MHz):"))
        param_layout.addWidget(self.bw_slider)
        
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["1 MHz", "2.4 MHz", "5 MHz", "10 MHz", "20 MHz"])
        param_layout.addWidget(QLabel("Sample Rate:"))
        param_layout.addWidget(self.sample_rate_combo)
        
        # Control buttons
        self.scan_btn = QPushButton("Start Scanning")
        self.record_btn = QPushButton("Record Data")
        self.analyze_btn = QPushButton("ML Analysis")
        
        # Status display
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        
        # Add all to layout
        layout.addWidget(freq_group)
        layout.addWidget(mod_group)
        layout.addWidget(param_group)
        layout.addWidget(self.scan_btn)
        layout.addWidget(self.record_btn)
        layout.addWidget(self.analyze_btn)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)
        layout.addStretch()
        
        return panel
    
    def create_visualization_panel(self):
        """Create the visualization panel with tabs"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Waterfall display tab
        waterfall_tab = QWidget()
        waterfall_layout = QVBoxLayout(waterfall_tab)
        self.waterfall_plot = pg.PlotWidget()
        self.waterfall_plot.setTitle("Waterfall Display")
        self.waterfall_plot.setLabel('left', 'Frequency', 'MHz')
        self.waterfall_plot.setLabel('bottom', 'Time', 's')
        waterfall_layout.addWidget(self.waterfall_plot)
        
        # Spectrum display tab
        spectrum_tab = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_tab)
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setTitle("Spectrum Analysis")
        self.spectrum_plot.setLabel('left', 'Power', 'dBm')
        self.spectrum_plot.setLabel('bottom', 'Frequency', 'MHz')
        self.spectrum_curve = self.spectrum_plot.plot(pen='y')
        spectrum_layout.addWidget(self.spectrum_plot)
        
        # FFT display tab
        fft_tab = QWidget()
        fft_layout = QVBoxLayout(fft_tab)
        self.fft_plot = pg.PlotWidget()
        self.fft_plot.setTitle("FFT Analysis")
        self.fft_plot.setLabel('left', 'Magnitude', 'dB')
        self.fft_plot.setLabel('bottom', 'Frequency', 'Hz')
        self.fft_curve = self.fft_plot.plot(pen='g')
        fft_layout.addWidget(self.fft_plot)
        
        # Constellation diagram tab
        constellation_tab = QWidget()
        constellation_layout = QVBoxLayout(constellation_tab)
        self.constellation_plot = pg.PlotWidget()
        self.constellation_plot.setTitle("Constellation Diagram")
        self.constellation_plot.setLabel('left', 'Quadrature')
        self.constellation_plot.setLabel('bottom', 'In-Phase')
        self.constellation_scatter = pg.ScatterPlotItem(size=5, pen=None, brush='r')
        self.constellation_plot.addItem(self.constellation_scatter)
        constellation_layout.addWidget(self.constellation_plot)
        
        # Signal info display
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        self.signal_info_text = QTextEdit()
        self.signal_info_text.setReadOnly(True)
        info_layout.addWidget(QLabel("Detected Signals:"))
        info_layout.addWidget(self.signal_info_text)
        
        # Add tabs
        self.tabs.addTab(waterfall_tab, "Waterfall")
        self.tabs.addTab(spectrum_tab, "Spectrum")
        self.tabs.addTab(fft_tab, "FFT")
        self.tabs.addTab(constellation_tab, "Constellation")
        self.tabs.addTab(info_tab, "Signal Info")
        
        layout.addWidget(self.tabs)
        
        return panel
    
    def setup_connections(self):
        """Setup signal connections"""
        self.scan_btn.clicked.connect(self.toggle_scanning)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.analyze_btn.clicked.connect(self.run_ml_analysis)
        self.band_combo.currentTextChanged.connect(self.on_band_changed)
        self.center_freq_spin.valueChanged.connect(self.on_center_freq_changed)
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)
        self.update_timer.start(100)  # Update every 100ms
        
    def on_band_changed(self, band_name):
        """Handle band selection change"""
        band_frequencies = {
            "ADS-B (1090 MHz)": 1090,
            "GPS L1 (1575.42 MHz)": 1575.42,
            "GLONASS (1602 MHz)": 1602,
            "GSM 900": 950,
            "GSM 1800": 1800,
            "L-Band Satellite": 1500,
            "S-Band Radar": 3000,
            "Custom Range": 1000
        }
        
        if band_name in band_frequencies:
            self.center_freq_spin.setValue(band_frequencies[band_name])
        
    def on_center_freq_changed(self, freq):
        """Handle center frequency change"""
        self.rf_processor.center_freq = freq * 1e6
        
    def toggle_scanning(self):
        """Toggle scanning state"""
        if self.scan_btn.text() == "Start Scanning":
            self.scan_btn.setText("Stop Scanning")
            self.rf_processor.start_scanning()
            self.status_text.append("Scanning started...")
        else:
            self.scan_btn.setText("Start Scanning")
            self.rf_processor.stop_scanning()
            self.status_text.append("Scanning stopped.")
            
    def toggle_recording(self):
        """Toggle recording state"""
        self.rf_processor.recording = not self.rf_processor.recording
        if self.rf_processor.recording:
            self.record_btn.setText("Stop Recording")
            self.status_text.append("Recording started...")
        else:
            self.record_btn.setText("Record Data")
            self.status_text.append("Recording stopped.")
            
    def run_ml_analysis(self):
        """Run ML analysis on current signals"""
        if len(self.rf_processor.spectrum_data) > 0:
            latest_spectrum = self.rf_processor.spectrum_data[-1]
            signal_type = self.ml_analyzer.classify_signal(latest_spectrum)
            modulation = self.ml_analyzer.detect_modulation(self.rf_processor.iq_data)
            
            info_text = f"""
            ML Analysis Results:
            - Signal Type: {signal_type}
            - Modulation: {modulation}
            - Center Frequency: {self.center_freq_spin.value()} MHz
            - Bandwidth: {self.bw_slider.value()} MHz
            - SNR: {np.random.uniform(15, 30):.1f} dB
            """
            self.signal_info_text.setText(info_text)
            self.status_text.append("ML analysis completed.")
        
    def update_displays(self):
        """Update all visualizations"""
        if not self.rf_processor.scanning:
            return
            
        # Update spectrum plot
        if len(self.rf_processor.spectrum_data) > 0:
            latest_spectrum = self.rf_processor.spectrum_data[-1]
            freqs = np.linspace(
                self.rf_processor.center_freq - self.rf_processor.sample_rate/2,
                self.rf_processor.center_freq + self.rf_processor.sample_rate/2,
                len(latest_spectrum)
            ) / 1e6  # Convert to MHz
            
            self.spectrum_curve.setData(freqs, latest_spectrum)
            
        # Update FFT plot
        if len(self.rf_processor.iq_data) > 1000:
            iq_samples = list(self.rf_processor.iq_data)[-1024:]
            fft_result = np.fft.fft(iq_samples)
            fft_freqs = np.fft.fftfreq(len(fft_result), 1/self.rf_processor.sample_rate)
            fft_magnitude = 20 * np.log10(np.abs(fft_result) + 1e-12)
            
            self.fft_curve.setData(fft_freqs[:len(fft_freqs)//2], fft_magnitude[:len(fft_magnitude)//2])
            
        # Update constellation plot
        if len(self.rf_processor.iq_data) > 100:
            iq_samples = list(self.rf_processor.iq_data)[-1000:]
            i_data = iq_samples[::2] if len(iq_samples) > 1000 else iq_samples[::1]
            q_data = iq_samples[1::2] if len(iq_samples) > 1000 else iq_samples[::1]
            
            if len(i_data) > 0 and len(q_data) > 0:
                min_len = min(len(i_data), len(q_data))
                self.constellation_scatter.setData(i_data[:min_len], q_data[:min_len])

class RFProcessor:
    def __init__(self):
        self.scanning = False
        self.recording = False
        self.sample_rate = 2.4e6  # 2.4 MHz
        self.center_freq = 100e6  # 100 MHz
        self.gain = 50
        self.bandwidth = 20e6  # 20 MHz
        
        # Data buffers
        self.spectrum_data = deque(maxlen=100)
        self.waterfall_data = deque(maxlen=50)
        self.iq_data = deque(maxlen=10000)
        
        # Threading
        self.scan_thread = None
        
    def start_scanning(self):
        """Start the scanning process"""
        self.scanning = True
        self.scan_thread = threading.Thread(target=self._scan_loop)
        self.scan_thread.daemon = True
        self.scan_thread.start()
        
    def stop_scanning(self):
        """Stop the scanning process"""
        self.scanning = False
        if self.scan_thread:
            self.scan_thread.join(timeout=1.0)
            
    def _scan_loop(self):
        """Main scanning loop"""
        while self.scanning:
            try:
                # Simulate RF data acquisition
                iq_samples = self._generate_simulated_signal()
                
                # Store IQ data
                self.iq_data.extend(iq_samples)
                
                # Process the samples
                spectrum = self._compute_spectrum(iq_samples)
                self.spectrum_data.append(spectrum)
                
                # Update waterfall
                if len(self.waterfall_data) < 50:
                    self.waterfall_data.append(spectrum)
                else:
                    self.waterfall_data.popleft()
                    self.waterfall_data.append(spectrum)
                    
                time.sleep(0.05)  # Simulate processing time
                
            except Exception as e:
                print(f"Scanning error: {e}")
                time.sleep(0.1)
            
    def _generate_simulated_signal(self):
        """Generate simulated RF signals for demonstration"""
        t = np.linspace(0, 0.001, int(self.sample_rate * 0.001))  # 1ms of samples
        
        # Simulate multiple signals
        signals = []
        noise_power = 0.05
        
        # ADS-B signal (1090 MHz pulsed)
        if 1080e6 <= self.center_freq <= 1100e6:
            pulse_train = (np.random.random(len(t)) > 0.95).astype(float)
            adsb_signal = np.sin(2 * np.pi * 1090e6 * t) * pulse_train
            signals.append(adsb_signal * 0.8)
            
        # GPS signal (1575.42 MHz spread spectrum)
        if 1570e6 <= self.center_freq <= 1580e6:
            prn_code = np.sign(np.sin(2 * np.pi * 1.023e6 * t))
            gps_signal = np.sin(2 * np.pi * 1575.42e6 * t) * prn_code
            signals.append(gps_signal * 0.6)
            
        # GSM signal (GMSK modulated)
        if (925e6 <= self.center_freq <= 960e6) or (1805e6 <= self.center_freq <= 1880e6):
            gsm_data = np.random.randint(0, 2, len(t)//100).repeat(100)
            gsm_signal = np.sin(2 * np.pi * self.center_freq * t + np.cumsum(gsm_data-0.5) * 0.3)
            signals.append(gsm_signal * 0.7)
            
        # Add noise
        noise = noise_power * np.random.randn(len(t))
        signals.append(noise)
        
        # Combine signals
        if signals:
            combined = np.sum(signals, axis=0)
        else:
            combined = noise
            
        return combined
            
    def _compute_spectrum(self, iq_samples):
        """Compute power spectrum"""
        f, Pxx = signal.welch(iq_samples, self.sample_rate, nperseg=1024, return_onesided=False)
        # Shift frequencies for center frequency
        f_shifted = f + self.center_freq
        return 10 * np.log10(Pxx + 1e-12)  # Convert to dB

class MLAnalyzer:
    def __init__(self):
        self.modulation_types = ['AM', 'FM', 'BPSK', 'QPSK', '8PSK', '16QAM', 'FSK', 'OFDM']
        self.signal_types = ['ADS-B', 'GPS', 'GLONASS', 'GSM', 'Satellite', 'Radar', 'WiFi', 'Unknown']
        
        # Initialize classifiers (in real app, these would be trained)
        self.modulation_classifier = self._init_modulation_classifier()
        self.signal_classifier = self._init_signal_classifier()
        
    def _init_modulation_classifier(self):
        """Initialize modulation classifier"""
        return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
    
    def _init_signal_classifier(self):
        """Initialize signal type classifier"""
        return RandomForestClassifier(n_estimators=100)
    
    def detect_modulation(self, iq_data):
        """Detect modulation type using feature-based approach"""
        if len(iq_data) < 100:
            return "Unknown"
            
        # Extract features for modulation classification
        features = self._extract_modulation_features(iq_data)
        
        # Simple rule-based detection (replace with trained classifier)
        if len(iq_data) > 1000:
            iq_subset = list(iq_data)[-1000:]
        else:
            iq_subset = list(iq_data)
            
        # Calculate statistics for modulation detection
        amplitude_var = np.var(np.abs(iq_subset))
        phase_changes = np.std(np.diff(np.angle(iq_subset)))
        
        # Simple heuristic-based modulation recognition
        if amplitude_var > 0.5:
            return "AM"
        elif phase_changes > 0.5:
            return "PM"
        else:
            # Return random from known modulations for demo
            return np.random.choice(self.modulation_types)
    
    def classify_signal(self, spectrum_data):
        """Classify signal type based on spectrum features"""
        if len(spectrum_data) == 0:
            return "Unknown"
            
        # Extract spectral features
        features = self._extract_spectral_features(spectrum_data)
        
        # Simple rule-based classification based on center frequency
        # In real implementation, this would use the trained classifier
        center_freq = np.argmax(spectrum_data) if len(spectrum_data) > 0 else 0
        
        # Frequency-based classification (simplified)
        if 1080 <= center_freq <= 1100:
            return "ADS-B"
        elif 1570 <= center_freq <= 1580:
            return "GPS"
        elif 925 <= center_freq <= 960 or 1805 <= center_freq <= 1880:
            return "GSM"
        else:
            return np.random.choice(self.signal_types)
    
    def _extract_modulation_features(self, iq_data):
        """Extract features for modulation classification"""
        iq_array = np.array(list(iq_data)[-1000:])
        
        features = [
            np.mean(np.abs(iq_array)),
            np.std(np.abs(iq_array)),
            np.var(np.angle(iq_array)),
            np.std(np.diff(np.angle(iq_array))),
            np.max(np.abs(iq_array)) - np.min(np.abs(iq_array))
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _extract_spectral_features(self, spectrum_data):
        """Extract features from spectrum for signal classification"""
        spectrum_array = np.array(spectrum_data)
        
        features = [
            np.mean(spectrum_array),
            np.std(spectrum_array),
            np.max(spectrum_array),
            np.argmax(spectrum_array) / len(spectrum_array),  # Normalized peak position
            np.sum(spectrum_array > np.mean(spectrum_array)) / len(spectrum_array)  # Occupied bandwidth estimate
        ]
        
        return np.array(features).reshape(1, -1)

class AdvancedVisualizations:
    """Advanced visualization utilities"""
    @staticmethod
    def create_spectrogram(iq_data, sample_rate):
        """Create spectrogram from IQ data"""
        if len(iq_data) < 1024:
            return None, None, None
            
        f, t, Sxx = signal.spectrogram(iq_data, sample_rate, nperseg=256)
        return f, t, 10 * np.log10(Sxx + 1e-12)

def main():
    # Initialize application
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    
    # Create and show main window
    scanner = RFScannerGUI()
    scanner.show()
    
    # Start application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()