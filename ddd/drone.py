#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defensive Drone Detection System for Security Research
Author: Security Research Team
License: For authorized defensive testing only
"""

import numpy as np
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class DefensiveDroneDetector:
    """
    LEGITIMATE DEFENSIVE DRONE DETECTION SYSTEM
    For authorized security research and protection purposes only
    """
    
    def __init__(self, frequency=868e6, sample_rate=2.4e6, gain=40):
        """Initialize SDR for drone signal detection"""
        self.sdr = None
        self.frequency = frequency  # Common drone control frequencies
        self.sample_rate = sample_rate
        self.gain = gain
        self.drone_signatures = {
            'DJI': [2.4e9, 5.8e9],
            'Parrot': [2.4835e9],
            'Autel': [2.4e9, 5.725e9]
        }
        
    def initialize_sdr(self):
        """Initialize RTL-SDR device"""
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.frequency
            self.sdr.gain = self.gain
            print(f"[+] SDR Initialized: {self.frequency/1e6} MHz")
            return True
        except Exception as e:
            print(f"[-] SDR Initialization Failed: {e}")
            return False
    
    def capture_samples(self, num_samples=256*1024):
        """Capture RF samples for analysis"""
        if not self.sdr:
            return None
        
        print(f"[*] Capturing {num_samples} samples...")
        samples = self.sdr.read_samples(num_samples)
        return samples
    
    def analyze_signals(self, samples):
        """Analyze captured signals for drone patterns"""
        if samples is None:
            return []
        
        # Perform FFT to analyze frequency spectrum
        fft_result = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(samples), 1/self.sample_rate)
        
        detected_drones = []
        
        # Check for known drone frequency signatures
        for drone_type, drone_freqs in self.drone_signatures.items():
            for target_freq in drone_freqs:
                if self.check_frequency_present(freqs, fft_result, target_freq):
                    detected_drones.append({
                        'type': drone_type,
                        'frequency': target_freq,
                        'strength': self.calculate_signal_strength(samples)
                    })
                    print(f"[!] Detected: {drone_type} at {target_freq/1e6} MHz")
        
        return detected_drones
    
    def check_frequency_present(self, freqs, fft_result, target_freq, threshold=0.1):
        """Check if specific frequency is present in signal"""
        idx = np.argmin(np.abs(freqs - target_freq))
        power = np.abs(fft_result[idx])
        avg_power = np.mean(np.abs(fft_result))
        
        return power > avg_power * threshold
    
    def calculate_signal_strength(self, samples):
        """Calculate signal strength in dB"""
        power = np.mean(np.abs(samples)**2)
        return 10 * np.log10(power) if power > 0 else -100
    
    def generate_alert(self, drone_info):
        """Generate defensive alert for detected drones"""
        alert_msg = f"""
        ================================
        DEFENSIVE ALERT - DRONE DETECTED
        ================================
        Type: {drone_info['type']}
        Frequency: {drone_info['frequency']/1e6} MHz
        Signal Strength: {drone_info['strength']:.2f} dB
        Timestamp: {np.datetime64('now')}
        
        [DEFENSIVE ACTION RECOMMENDED]
        1. Verify authorization status
        2. Document detection
        3. Notify security personnel
        4. Follow local regulations
        ================================
        """
        print(alert_msg)
        return alert_msg
    
    def defensive_analysis_report(self, detected_drones):
        """Generate defensive analysis report"""
        report = f"""
        DEFENSIVE DRONE DETECTION REPORT
        =================================
        Scan Frequency: {self.frequency/1e6} MHz
        Sample Rate: {self.sample_rate/1e6} MS/s
        Detected Threats: {len(detected_drones)}
        
        DETECTED DEVICES:
        """
        
        for i, drone in enumerate(detected_drones, 1):
            report += f"""
        {i}. {drone['type']}:
           - Frequency: {drone['frequency']/1e6} MHz
           - Strength: {drone['strength']:.2f} dB
           - Risk Level: MEDIUM
           - Recommended Response: Monitor and document
            """
        
        report += f"""
        
        DEFENSIVE RECOMMENDATIONS:
        1. Legal Compliance Check
        2. Authorization Verification
        3. Documentation for Security Audit
        4. Regular System Updates
        5. Personnel Training
        
        IMPORTANT LEGAL NOTICE:
        This system is for authorized defensive security research only.
        Unauthorized drone interception or disruption is illegal.
        Always comply with local regulations (FCC, ETSI, etc.).
        """
        
        print(report)
        return report
    
    def cleanup(self):
        """Cleanup resources"""
        if self.sdr:
            self.sdr.close()
            print("[*] SDR resources released")

def main():
    """Main defensive monitoring function"""
    print("""
    ===========================================
    DEFENSIVE DRONE DETECTION SYSTEM - v1.0
    Authorized Security Research Only
    ===========================================
    """)
    
    # Initialize detector for common ISM bands
    detector = DefensiveDroneDetector(
        frequency=2.4e9,  # 2.4 GHz ISM band
        sample_rate=2.4e6,
        gain=40
    )
    
    if not detector.initialize_sdr():
        return
    
    try:
        # Capture and analyze signals
        samples = detector.capture_samples(1024*1024)
        detected_drones = detector.analyze_signals(samples)
        
        # Generate alerts and reports
        for drone in detected_drones:
            detector.generate_alert(drone)
        
        # Complete defensive report
        if detected_drones:
            detector.defensive_analysis_report(detected_drones)
        else:
            print("[*] No drone signals detected in monitored spectrum")
        
    except KeyboardInterrupt:
        print("\n[*] Monitoring stopped by user")
    except Exception as e:
        print(f"[-] Error: {e}")
    finally:
        detector.cleanup()
        print("[*] Defensive monitoring completed")

if __name__ == "__main__":
    # LEGAL WARNING: This tool is for authorized security research only
    # Violating laws regarding RF spectrum usage is a criminal offense
    # Always obtain proper authorization before use
    
    main()
