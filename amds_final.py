#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# amds_final.py - نظام كشف الصواريخ المتقدم
# نظام كامل يعمل على Kali Linux بدون مشاكل استيراد

"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║    █████╗ ███╗   ███╗██████╗ ███████╗    نظام كشف الصواريخ المتقدم                      ║
║   ██╔══██╗████╗ ████║██╔══██╗██╔════╝    Advanced Missile Detection System       ║
║   ███████║██╔████╔██║██║  ██║███████╗    الإصدار 2026 IRAN - ملف واحد متكامل           ║
║   ██╔══██║██║╚██╔╝██║██║  ██║╚════██║    للأغراض العسكرية والبحثية فقط                   ║
║   ██║  ██║██║ ╚═╝ ██║██████╔╝███████║    تطوير: خلية شرار تقدمها                       ║
║   ╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚══════╝    الى الجمهورية الاسلامية الايرانية                   ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import time
import random
import math
import json
from datetime import datetime
from enum import Enum
from collections import deque
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

# ============================================
# الجزء 1: تعريف الأنظمة والهياكل
# ============================================

class ThreatLevel(Enum):
    """مستويات التهديد"""
    SAFE = "آمن"
    LOW = "منخفض"
    MEDIUM = "متوسط"
    HIGH = "عالي"
    CRITICAL = "حرج"

class RadarMode(Enum):
    """أنماط عمل الرادار"""
    SEARCH = "بحث"
    TRACK = "تتبع"
    TRACK_WHILE_SCAN = "بحث وتتبع"
    ILLUMINATOR = "إضاءة"

class MissileType(Enum):
    """أنواع الصواريخ"""
    BALLISTIC = "بالستي"
    CRUISE = "كروز"
    AIR_TO_AIR = "جو-جو"
    SURFACE_TO_AIR = "أرض-جو"
    ANTI_SHIP = "ضد السفن"
    UNKNOWN = "غير معروف"

@dataclass
class Target:
    """هيكل بيانات الهدف"""
    id: str
    position: np.ndarray  # [x, y, z] بالأمتار
    velocity: np.ndarray  # [vx, vy, vz] م/ث
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radar_cross_section: float = 1.0  # RCS بالمتر المربع
    missile_type: MissileType = MissileType.UNKNOWN
    threat_level: ThreatLevel = ThreatLevel.LOW
    detection_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0  # ثقة الكشف (0-100%)
    trajectory: List[np.ndarray] = field(default_factory=list)
    is_hostile: bool = False
    
    def __post_init__(self):
        """تهيئة إضافية بعد الإنشاء"""
        if np.linalg.norm(self.velocity) > 1000:  # إذا كانت السرعة > 1000 م/ث
            self.missile_type = MissileType.BALLISTIC
            self.threat_level = ThreatLevel.CRITICAL
            self.is_hostile = True
    
    def update_position(self, dt: float = 1.0):
        """تحديث موقع الهدف بناءً على السرعة والتسارع"""
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2
        self.velocity += self.acceleration * dt
        self.last_update = datetime.now()
        
        # حفظ المسار (آخر 50 نقطة)
        self.trajectory.append(self.position.copy())
        if len(self.trajectory) > 50:
            self.trajectory.pop(0)

@dataclass
class RadarParameters:
    """معلمات الرادار"""
    name: str = "AN/SPY-6 AMDR"
    frequency: float = 10e9  # 10 GHz
    wavelength: float = field(init=False)
    power: float = 1000e3  # 1 ميجاوات
    peak_power: float = 10e6  # 10 ميجاوات
    pulse_width: float = 100e-6  # 100 ميكروثانية
    prf: float = 3000  # هرتز
    bandwidth: float = 10e6  # 10 MHz
    range_resolution: float = field(init=False)
    max_range: float = 500e3  # 500 كم
    min_range: float = 100  # 100 متر
    beam_width: float = 1.5  # درجات
    antenna_gain: float = 40  # ديسيبل
    noise_figure: float = 3.0  # ديسيبل
    system_losses: float = 10.0  # ديسيبل
    
    def __post_init__(self):
        """حساب القيم المشتقة"""
        self.wavelength = 3e8 / self.frequency
        self.range_resolution = 3e8 / (2 * self.bandwidth)
    
    def calculate_range_equation(self, target_rcs: float, range_km: float) -> float:
        """معادلة المدى الراداري"""
        # R^4 = (Pt * G^2 * λ^2 * σ) / ((4π)^3 * Pr * Ls)
        range_m = range_km * 1000
        
        # حساب القدرة المستلمة
        numerator = (self.peak_power * (10**(self.antenna_gain/10))**2 * 
                    self.wavelength**2 * target_rcs)
        denominator = ((4 * math.pi)**3 * range_m**4 * 
                      10**(self.system_losses/10))
        
        received_power = numerator / denominator
        return received_power

# ============================================
# الجزء 2: معالجة الإشارات الرادارية
# ============================================

class SignalProcessor:
    """معالج الإشارات الرادارية"""
    
    @staticmethod
    def generate_chirp_signal(duration: float, bandwidth: float, 
                             sampling_rate: float) -> np.ndarray:
        """توليد إشارة خطية التردد (Chirp)"""
        t = np.linspace(0, duration, int(duration * sampling_rate))
        chirp_rate = bandwidth / duration
        phase = 2 * np.pi * (chirp_rate/2 * t**2)
        signal = np.exp(1j * phase)
        return signal
    
    @staticmethod
    def apply_range_compression(signal: np.ndarray, 
                               reference_chirp: np.ndarray) -> np.ndarray:
        """ضغط المدى باستخدام الارتباط"""
        compressed = np.correlate(signal, reference_chirp, mode='same')
        return compressed
    
    @staticmethod
    def apply_pulse_compression(pulses: np.ndarray) -> np.ndarray:
        """ضغط النبضات Doppler"""
        doppler_profile = np.fft.fft(pulses, axis=0)
        return np.abs(doppler_profile)
    
    @staticmethod
    def cfar_detection(signal: np.ndarray, guard_cells: int = 2,
                      reference_cells: int = 10, pfa: float = 1e-6) -> np.ndarray:
        """خوارزمية CFAR للكشف التكيفي"""
        n = len(signal)
        threshold = np.zeros(n)
        detections = np.zeros(n, dtype=bool)
        
        for i in range(n):
            start_left = max(0, i - reference_cells - guard_cells)
            end_left = max(0, i - guard_cells)
            
            start_right = min(n, i + guard_cells + 1)
            end_right = min(n, i + guard_cells + reference_cells + 1)
            
            # الخلايا المرجعية
            reference_window = np.concatenate([
                signal[start_left:end_left],
                signal[start_right:end_right]
            ])
            
            if len(reference_window) > 0:
                # حساب العتبة
                noise_estimate = np.mean(reference_window)
                threshold_factor = -np.log(pfa)
                threshold[i] = noise_estimate * threshold_factor
                
                # الكشف
                if signal[i] > threshold[i]:
                    detections[i] = True
        
        return detections, threshold

# ============================================
# الجزء 3: مرشحات التتبع
# ============================================

class KalmanFilter:
    """مرشح كالمان لتتبع الأهداف"""
    
    def __init__(self, dim_x: int = 6, dim_z: int = 3):
        self.dim_x = dim_x  # أبعاد الحالة
        self.dim_z = dim_z  # أبعاد القياس
        
        # مصفوفة الحالة [x, y, z, vx, vy, vz]
        self.x = np.zeros(dim_x)
        
        # مصفوفة التغاير
        self.P = np.eye(dim_x) * 1000
        
        # مصفوفة الانتقال
        self.F = np.eye(dim_x)
        
        # مصفوفة القياس
        self.H = np.zeros((dim_z, dim_x))
        self.H[:dim_z, :dim_z] = np.eye(dim_z)
        
        # ضوضاء العملية
        self.Q = np.eye(dim_x) * 0.1
        
        # ضوضاء القياس
        self.R = np.eye(dim_z) * 1.0
        
        # كالمان غين
        self.K = np.zeros((dim_x, dim_z))
        
        self.last_prediction = datetime.now()
    
    def predict(self, dt: float = None):
        """توقع الحالة التالية"""
        if dt is None:
            current_time = datetime.now()
            dt = (current_time - self.last_prediction).total_seconds()
            self.last_prediction = current_time
        
        # تحديث مصفوفة الانتقال
        self.F[:3, 3:] = np.eye(3) * dt
        
        # توقع الحالة
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[:3]  # إرجاع الموقع فقط
    
    def update(self, z: np.ndarray):
        """تحديث المرشح بالقياسات الجديدة"""
        # الابتكار (الفرق بين القياس والتوقع)
        y = z - self.H @ self.x
        
        # مصفوفة الابتكار
        S = self.H @ self.P @ self.H.T + self.R
        
        # كالمان غين
        self.K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # تحديث الحالة
        self.x = self.x + self.K @ y
        
        # تحديث التباين
        I = np.eye(self.dim_x)
        self.P = (I - self.K @ self.H) @ self.P
        
        return self.x
    
    def get_estimated_state(self):
        """الحصول على الحالة المقدرة"""
        return {
            'position': self.x[:3],
            'velocity': self.x[3:6],
            'covariance': self.P[:3, :3]
        }

# ============================================
# الجزء 4: نظام الرادار المتقدم
# ============================================

class AdvancedRadarSystem:
    """النظام الرئيسي للرادار المتقدم"""
    
    def __init__(self):
        print("🚀 جاري تهيئة نظام الرادار المتقدم...")
        
        # معلمات الرادار
        self.params = RadarParameters()
        
        # أنظمة فرعية
        self.signal_processor = SignalProcessor()
        self.track_filters: Dict[str, KalmanFilter] = {}
        
        # بيانات النظام
        self.targets: Dict[str, Target] = {}
        self.detection_history = deque(maxlen=1000)
        self.threat_assessment = ThreatLevel.SAFE
        
        # حالة النظام
        self.is_active = False
        self.current_mode = RadarMode.SEARCH
        self.scan_angle = 0.0
        self.scan_elevation = 0.0
        
        # إحصائيات
        self.stats = {
            'total_scans': 0,
            'targets_detected': 0,
            'missiles_identified': 0,
            'false_alarms': 0,
            'scan_rate': 0,
            'start_time': datetime.now()
        }
        
        # خيوط المعالجة
        self.radar_thread = None
        self.display_thread = None
        
        # تهيئة واجهة العرض
        self.init_display()
        
        print("✅ تم تهيئة النظام بنجاح!")
        print(f"📡 نظام الرادار: {self.params.name}")
        print(f"🎯 المدى الأقصى: {self.params.max_range/1000:.0f} كم")
        print(f"📶 التردد: {self.params.frequency/1e9:.1f} GHz")
    
    def init_display(self):
        """تهيئة نظام العرض"""
        self.display_data = {
            'range_profile': None,
            'doppler_profile': None,
            'detected_targets': [],
            'threat_level': ThreatLevel.SAFE,
            'system_status': 'متوقف'
        }
    
    def start(self):
        """بدء تشغيل النظام"""
        if self.is_active:
            print("⚠️  النظام يعمل بالفعل")
            return False
        
        print("🔍 بدء تشغيل الرادار...")
        self.is_active = True
        self.display_data['system_status'] = 'نشط'
        
        # بدء خيوط المعالجة
        self.radar_thread = threading.Thread(target=self.radar_operation_cycle, daemon=True)
        self.display_thread = threading.Thread(target=self.display_update_cycle, daemon=True)
        
        self.radar_thread.start()
        self.display_thread.start()
        
        print("✅ بدأ تشغيل النظام")
        return True
    
    def stop(self):
        """إيقاف النظام"""
        if not self.is_active:
            print("⚠️  النظام متوقف بالفعل")
            return False
        
        print("⏹️ إيقاف النظام...")
        self.is_active = False
        
        if self.radar_thread:
            self.radar_thread.join(timeout=2)
        if self.display_thread:
            self.display_thread.join(timeout=2)
        
        self.display_data['system_status'] = 'متوقف'
        print("✅ تم إيقاف النظام")
        return True
    
    def radar_operation_cycle(self):
        """دورة عمل الرادار الرئيسية"""
        scan_counter = 0
        
        while self.is_active:
            try:
                scan_counter += 1
                self.stats['total_scans'] += 1
                
                # تغيير زاوية المسح
                self.scan_angle = (self.scan_angle + 1.5) % 360
                self.scan_elevation = 30 * math.sin(scan_counter * 0.1)
                
                # توليد إشارة رادارية
                chirp_signal = self.generate_radar_pulse()
                
                # محاكاة البيئة والأهداف
                environment_response = self.simulate_environment(chirp_signal)
                
                # معالجة الإشارة المستلمة
                processed_data = self.process_received_signal(environment_response)
                
                # كشف الأهداف
                detected_points = self.detect_targets(processed_data)
                
                # تتبع وتحديث الأهداف
                self.update_target_tracking(detected_points)
                
                # تقييم التهديد
                self.assess_threat_level()
                
                # تحديث الإحصائيات
                self.update_statistics()
                
                # تسجيل البيانات
                self.log_detection_data(detected_points)
                
                # حساب معدل المسح
                if scan_counter % 10 == 0:
                    elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
                    self.stats['scan_rate'] = scan_counter / elapsed if elapsed > 0 else 0
                
                # انتظار بين الدورات (معدل تحديث 10 هرتز)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️  خطأ في دورة الرادار: {e}")
                time.sleep(1)
    
    def generate_radar_pulse(self) -> np.ndarray:
        """توليد نبضة رادارية"""
        # توليد إشارة Chirp
        chirp = self.signal_processor.generate_chirp_signal(
            duration=self.params.pulse_width,
            bandwidth=self.params.bandwidth,
            sampling_rate=2 * self.params.bandwidth
        )
        
        # تطبيق خصائص الهوائي
        antenna_pattern = self.calculate_antenna_pattern()
        pulse = chirp * antenna_pattern * np.sqrt(self.params.peak_power)
        
        return pulse
    
    def calculate_antenna_pattern(self) -> float:
        """حساب نمط الهوائي الحالي"""
        # نمط بسيط للهوائي
        beam_width_rad = math.radians(self.params.beam_width)
        angle_diff = math.radians(self.scan_angle)
        
        # توزيع غوسي
        gain = math.exp(-(angle_diff**2) / (2 * (beam_width_rad/2)**2))
        return gain
    
    def simulate_environment(self, transmitted_signal: np.ndarray) -> Dict:
        """محاكاة البيئة والأهداف"""
        # إضافة ضوضاء
        noise_power = self.calculate_noise_power()
        noise = np.random.normal(0, np.sqrt(noise_power/2), len(transmitted_signal))
        noise = noise + 1j * noise  # ضوضاء مركبة
        
        # إشارة مستلمة (تبدأ بالإشارة المرسلة + ضوضاء)
        received_signal = transmitted_signal + noise
        
        # محاكاة أهداف عشوائية
        targets_response = np.zeros_like(transmitted_signal, dtype=complex)
        
        # توليد أهداف محاكاة
        self.generate_simulated_targets()
        
        # إضافة استجابات الأهداف
        for target_id, target in list(self.targets.items()):
            # حساب التأخير الزمني
            range_distance = np.linalg.norm(target.position)
            time_delay = 2 * range_distance / 3e8
            
            # حساب خسائر المسار
            wavelength = 3e8 / self.params.frequency
            path_loss = (wavelength**2 * target.radar_cross_section) / \
                       ((4 * math.pi)**3 * range_distance**4)
            
            # تأثير دوبلر
            radial_velocity = np.dot(target.velocity, 
                                   -target.position/np.linalg.norm(target.position))
            doppler_shift = 2 * radial_velocity / wavelength
            
            # إنشاء إشارة الهدف
            t = np.linspace(0, self.params.pulse_width, len(transmitted_signal))
            target_signal = np.sqrt(path_loss) * \
                          transmitted_signal * \
                          np.exp(1j * 2 * math.pi * doppler_shift * t)
            
            # تطبيق التأخير
            delay_samples = int(time_delay * 2 * self.params.bandwidth)
            if delay_samples < len(target_signal):
                target_signal = np.roll(target_signal, delay_samples)
                target_signal[:delay_samples] = 0
            
            targets_response += target_signal
        
        received_signal += targets_response
        
        return {
            'signal': received_signal,
            'noise_power': noise_power,
            'timestamp': datetime.now(),
            'scan_angle': self.scan_angle,
            'scan_elevation': self.scan_elevation
        }
    
    def calculate_noise_power(self) -> float:
        """حساب قدرة الضوضاء"""
        # Pn = k * T * B * F
        k = 1.38e-23  # بولتزمان
        T = 290  # درجة الحرارة بالكلفن
        B = self.params.bandwidth
        F = 10**(self.params.noise_figure/10)
        
        noise_power = k * T * B * F
        return noise_power
    
    def generate_simulated_targets(self):
        """توليد أهداف محاكاة عشوائية"""
        # فرصة إضافة هدف جديد
        if random.random() < 0.1 and len(self.targets) < 20:  # 10% فرصة
            self.create_random_target()
        
        # تحديث مواقع الأهداف الحالية
        for target_id, target in list(self.targets.items()):
            # تحديث الموقع
            dt = (datetime.now() - target.last_update).total_seconds()
            target.update_position(dt)
            
            # حذف الأهداف خارج النطاق
            if np.linalg.norm(target.position) > self.params.max_range:
                del self.targets[target_id]
                if target_id in self.track_filters:
                    del self.track_filters[target_id]
    
    def create_random_target(self):
        """إنشاء هدف عشوائي"""
        target_id = f"TGT-{random.randint(10000, 99999)}"
        
        # توليد موقع عشوائي
        range_km = random.uniform(10, 400)
        azimuth = random.uniform(0, 360)
        elevation = random.uniform(0, 45)
        
        # تحويل إلى إحداثيات ديكارتية
        range_m = range_km * 1000
        azimuth_rad = math.radians(azimuth)
        elevation_rad = math.radians(elevation)
        
        x = range_m * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = range_m * math.cos(elevation_rad) * math.sin(azimuth_rad)
        z = range_m * math.sin(elevation_rad)
        
        # سرعة عشوائية
        speed = random.uniform(100, 2500)
        heading = random.uniform(0, 360)
        climb = random.uniform(-10, 10)
        
        heading_rad = math.radians(heading)
        climb_rad = math.radians(climb)
        
        vx = speed * math.cos(climb_rad) * math.cos(heading_rad)
        vy = speed * math.cos(climb_rad) * math.sin(heading_rad)
        vz = speed * math.sin(climb_rad)
        
        # تحديد نوع الهدف بناءً على السرعة
        if speed > 2000:
            missile_type = MissileType.BALLISTIC
            threat_level = ThreatLevel.CRITICAL
            is_hostile = True
        elif speed > 800:
            missile_type = MissileType.CRUISE
            threat_level = ThreatLevel.HIGH
            is_hostile = True
        else:
            missile_type = MissileType.UNKNOWN
            threat_level = ThreatLevel.LOW
            is_hostile = random.random() < 0.3  # 30% فرصة أن يكون عدائياً
        
        # إنشاء الهدف
        target = Target(
            id=target_id,
            position=np.array([x, y, z]),
            velocity=np.array([vx, vy, vz]),
            missile_type=missile_type,
            threat_level=threat_level,
            radar_cross_section=random.uniform(0.01, 10.0),
            confidence=random.uniform(70, 95),
            is_hostile=is_hostile
        )
        
        self.targets[target_id] = target
        
        # إنشاء مرشح كالمان جديد
        self.track_filters[target_id] = KalmanFilter()
        self.track_filters[target_id].x[:3] = target.position
        self.track_filters[target_id].x[3:6] = target.velocity
        
        return target_id
    
    def process_received_signal(self, environment_data: Dict) -> Dict:
        """معالجة الإشارة المستلمة"""
        signal = environment_data['signal']
        
        # ضغط المدى
        reference_chirp = self.signal_processor.generate_chirp_signal(
            self.params.pulse_width,
            self.params.bandwidth,
            2 * self.params.bandwidth
        )
        
        range_compressed = self.signal_processor.apply_range_compression(
            signal, reference_chirp
        )
        
        # تحليل دوبلر (محاكاة)
        num_pulses = 32
        doppler_data = np.zeros((num_pulses, len(range_compressed)), dtype=complex)
        
        for i in range(num_pulses):
            doppler_data[i, :] = range_compressed * \
                               np.exp(1j * 2 * math.pi * i / num_pulses)
        
        doppler_profile = self.signal_processor.apply_pulse_compression(doppler_data)
        
        return {
            'range_profile': np.abs(range_compressed),
            'doppler_profile': doppler_profile,
            'timestamp': environment_data['timestamp'],
            'scan_angle': environment_data['scan_angle'],
            'scan_elevation': environment_data['scan_elevation']
        }
    
    def detect_targets(self, processed_data: Dict) -> List[Dict]:
        """كشف الأهداف من البيانات المعالجة"""
        range_profile = processed_data['range_profile']
        
        # تطبيق CFAR
        detections, threshold = self.signal_processor.cfar_detection(
            range_profile, pfa=1e-5
        )
        
        # إيجاد الذروات
        detection_indices = np.where(detections)[0]
        
        # تحويل المؤشرات إلى معلومات هدف
        detected_points = []
        range_bin_size = self.params.range_resolution
        
        for idx in detection_indices:
            range_distance = idx * range_bin_size
            
            # تجاهل الأهداف خارج النطاق
            if range_distance > self.params.max_range:
                continue
            
            # إنشاء نقطة كشف
            point = {
                'range': range_distance,
                'angle': processed_data['scan_angle'],
                'elevation': processed_data['scan_elevation'],
                'snr': range_profile[idx] / np.mean(threshold),
                'timestamp': processed_data['timestamp']
            }
            
            detected_points.append(point)
            
            # تحديث الإحصائيات
            self.stats['targets_detected'] += 1
        
        # تحديث بيانات العرض
        self.display_data['range_profile'] = range_profile
        self.display_data['doppler_profile'] = processed_data['doppler_profile']
        self.display_data['detected_targets'] = detected_points
        
        return detected_points
    
    def update_target_tracking(self, detected_points: List[Dict]):
        """تحديث تتبع الأهداف"""
        for point in detected_points:
            # تحويل الإحداثيات الكروية إلى ديكارتية
            range_m = point['range']
            azimuth_rad = math.radians(point['angle'])
            elevation_rad = math.radians(point['elevation'])
            
            x = range_m * math.cos(elevation_rad) * math.cos(azimuth_rad)
            y = range_m * math.cos(elevation_rad) * math.sin(azimuth_rad)
            z = range_m * math.sin(elevation_rad)
            
            measurement = np.array([x, y, z])
            
            # البحث عن هدف قريب للمطابقة
            matched_target_id = self.find_matching_target(measurement)
            
            if matched_target_id:
                # تحديث الهدف الموجود
                target = self.targets[matched_target_id]
                kalman_filter = self.track_filters[matched_target_id]
                
                # تحديث مرشح كالمان
                kalman_filter.predict()
                estimated_state = kalman_filter.update(measurement)
                
                # تحديث بيانات الهدف
                target.position = estimated_state[:3]
                target.velocity = estimated_state[3:6]
                target.confidence = min(100, target.confidence + 5)
                target.last_update = datetime.now()
                
            else:
                # إنشاء هدف جديد للكشف القوي
                if point['snr'] > 20:  # عتبة SNR عالية للكشف الجديد
                    target_id = self.create_target_from_detection(point)
                    self.stats['missiles_identified'] += 1
    
    def find_matching_target(self, measurement: np.ndarray, 
                           max_distance: float = 5000) -> Optional[str]:
        """البحث عن هدف قريب للمطابقة"""
        for target_id, target in self.targets.items():
            distance = np.linalg.norm(target.position - measurement)
            if distance < max_distance:
                return target_id
        return None
    
    def create_target_from_detection(self, detection: Dict) -> str:
        """إنشاء هدف جديد من نقطة كشف"""
        target_id = f"DET-{random.randint(10000, 99999)}"
        
        # تحويل الإحداثيات
        range_m = detection['range']
        azimuth_rad = math.radians(detection['angle'])
        elevation_rad = math.radians(detection['elevation'])
        
        x = range_m * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = range_m * math.cos(elevation_rad) * math.sin(azimuth_rad)
        z = range_m * math.sin(elevation_rad)
        
        # سرعة افتراضية
        speed = 300  # م/ث
        velocity = np.array([
            speed * random.uniform(-0.5, 0.5),
            speed * random.uniform(-0.5, 0.5),
            speed * random.uniform(-0.1, 0.1)
        ])
        
        # إنشاء الهدف
        target = Target(
            id=target_id,
            position=np.array([x, y, z]),
            velocity=velocity,
            confidence=detection['snr'] * 5,  # تحويل SNR إلى ثقة
            is_hostile=detection['snr'] > 15
        )
        
        self.targets[target_id] = target
        
        # إنشاء مرشح كالمان
        self.track_filters[target_id] = KalmanFilter()
        self.track_filters[target_id].x[:3] = target.position
        self.track_filters[target_id].x[3:6] = target.velocity
        
        return target_id
    
    def assess_threat_level(self):
        """تقييم مستوى التهديد العام"""
        if not self.targets:
            self.threat_assessment = ThreatLevel.SAFE
            self.display_data['threat_level'] = ThreatLevel.SAFE
            return
        
        # حساب أعلى مستوى تهديد
        max_threat = max((t.threat_level for t in self.targets.values()), 
                        key=lambda x: x.value)
        
        # تحسين التقييم بناءً على عدد الأهداف
        hostile_count = sum(1 for t in self.targets.values() if t.is_hostile)
        
        if hostile_count >= 5:
            self.threat_assessment = ThreatLevel.CRITICAL
        elif hostile_count >= 3:
            self.threat_assessment = ThreatLevel.HIGH
        elif hostile_count >= 1:
            self.threat_assessment = ThreatLevel.MEDIUM
        else:
            self.threat_assessment = max_threat
        
        self.display_data['threat_level'] = self.threat_assessment
    
    def update_statistics(self):
        """تحديث إحصائيات النظام"""
        # تحديث معدل الاكتشاف
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        if elapsed > 0:
            self.stats['scan_rate'] = self.stats['total_scans'] / elapsed
    
    def log_detection_data(self, detected_points: List[Dict]):
        """تسجيل بيانات الكشف"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'targets_count': len(self.targets),
            'detections': len(detected_points),
            'threat_level': self.threat_assessment.value,
            'hostile_targets': sum(1 for t in self.targets.values() if t.is_hostile)
        }
        
        self.detection_history.append(log_entry)
    
    def display_update_cycle(self):
        """دورة تحديث العرض"""
        while self.is_active:
            try:
                self.display_radar_screen()
                time.sleep(0.5)  # تحديث كل 0.5 ثانية
            except:
                time.sleep(1)
    
    def display_radar_screen(self):
        """عرض شاشة الرادار في الطرفية"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║                    شاشة الرادار - نظام AMDS                     ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        
        # حالة النظام
        status_color = "🟢" if self.is_active else "🔴"
        print(f"║  الحالة: {status_color} {self.display_data['system_status']:40} ║")
        
        # مستوى التهديد
        threat_icon = {
            ThreatLevel.SAFE: "🟢",
            ThreatLevel.LOW: "🟡",
            ThreatLevel.MEDIUM: "🟠",
            ThreatLevel.HIGH: "🔴",
            ThreatLevel.CRITICAL: "💀"
        }.get(self.threat_assessment, "⚪")
        
        print(f"║  مستوى التهديد: {threat_icon} {self.threat_assessment.value:36} ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        
        # إحصائيات
        print(f"║  الأهداف: {len(self.targets):3d} ║ المسوح: {self.stats['total_scans']:6d} ║")
        print(f"║  الصواريخ: {self.stats['missiles_identified']:3d} ║ المعدل: {self.stats['scan_rate']:6.1f}Hz ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        
        # قائمة الأهداف
        print("║                         الأهداف المكتشفة                        ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        
        if not self.targets:
            print("║                          لا توجد أهداف                         ║")
        else:
            target_list = list(self.targets.values())[:8]  # عرض أول 8 أهداف فقط
            
            for i, target in enumerate(target_list):
                range_km = np.linalg.norm(target.position) / 1000
                speed = np.linalg.norm(target.velocity)
                
                # أيقونة الهدف
                icon = "🚀" if target.is_hostile else "✈️"
                if target.missile_type == MissileType.BALLISTIC:
                    icon = "🛰️"
                
                # لون التهديد
                threat_color = {
                    ThreatLevel.SAFE: "🟢",
                    ThreatLevel.LOW: "🟡",
                    ThreatLevel.MEDIUM: "🟠",
                    ThreatLevel.HIGH: "🔴",
                    ThreatLevel.CRITICAL: "💀"
                }.get(target.threat_level, "⚪")
                
                line = f"  {icon} {target.id} | {range_km:5.1f}km | {speed:5.0f}m/s | {threat_color}"
                print(f"║{line:58}║")
        
        print("╠══════════════════════════════════════════════════════════════════╣")
        
        # معلومات الرادار
        print(f"║  زاوية المسح: {self.scan_angle:6.1f}° ║ الارتفاع: {self.scan_elevation:6.1f}° ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        
        # تعليمات
        print("\n🎮 تعليمات: [S] بدء/إيقاف | [Q] خروج | [A] إضافة هدف | [C] مسح الشاشة")
    
    def get_system_info(self) -> Dict:
        """الحصول على معلومات النظام"""
        runtime = datetime.now() - self.stats['start_time']
        
        return {
            'runtime': str(runtime).split('.')[0],
            'radar_system': self.params.name,
            'status': 'نشط' if self.is_active else 'متوقف',
            'mode': self.current_mode.value,
            'targets_count': len(self.targets),
            'hostile_targets': sum(1 for t in self.targets.values() if t.is_hostile),
            'threat_level': self.threat_assessment.value,
            'statistics': self.stats.copy()
        }

# ============================================
# الجزء 5: الواجهة الرئيسية
# ============================================

class AMDSInterface:
    """واجهة المستخدم للنظام"""
    
    def __init__(self):
        self.radar_system = AdvancedRadarSystem()
        self.running = True
    
    def display_banner(self):
        """عرض شعار النظام"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║    █████╗ ███╗   ███╗██████╗ ███████╗    نظام كشف الصواريخ المتقدم                      ║
║   ██╔══██╗████╗ ████║██╔══██╗██╔════╝    Advanced Missile Detection System       ║
║   ███████║██╔████╔██║██║  ██║███████╗    الإصدار 1.0                               ║
║   ██╔══██║██║╚██╔╝██║██║  ██║╚════██║    للأغراض العسكرية والبحثية فقط                    ║
║   ██║  ██║██║ ╚═╝ ██║██████╔╝███████║    تطوير: خلية شرار الاستخبارية تقدمها                ║
║   ╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚══════╝    الى الجمهورية الاسلامية الايرانية                   ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def display_main_menu(self):
        """عرض القائمة الرئيسية"""
        print("\n" + "═" * 70)
        print("📋 القائمة الرئيسية - نظام كشف الصواريخ المتقدم (AMDS)")
        print("═" * 70)
        print("1. 🚀  بدء تشغيل النظام")
        print("2. ⏹️  إيقاف النظام")
        print("3. 📊  عرض معلومات النظام")
        print("4. 🎯  عرض الأهداف الحالية")
        print("5. 🎮  محاكاة سريعة (60 ثانية)")
        print("6. ⚙️   إعدادات الرادار")
        print("7. 📈  عرض الإحصائيات")
        print("8. 💾  حفظ بيانات النظام")
        print("9. 🆘  المساعدة")
        print("0. ❌  خروج")
        print("═" * 70)
    
    def handle_user_input(self):
        """معالجة إدخال المستخدم"""
        try:
            choice = input("\n📝 اختر خياراً (0-9): ").strip()
            
            if choice == '1':
                self.start_system()
            elif choice == '2':
                self.stop_system()
            elif choice == '3':
                self.show_system_info()
            elif choice == '4':
                self.show_targets()
            elif choice == '5':
                self.run_quick_simulation()
            elif choice == '6':
                self.show_radar_settings()
            elif choice == '7':
                self.show_statistics()
            elif choice == '8':
                self.save_system_data()
            elif choice == '9':
                self.show_help()
            elif choice == '0':
                self.exit_system()
            elif choice.lower() == 's':
                # اختصار لبدء/إيقاف
                if self.radar_system.is_active:
                    self.stop_system()
                else:
                    self.start_system()
            elif choice.lower() == 'a':
                self.add_random_target()
            elif choice.lower() == 'c':
                os.system('clear' if os.name == 'posix' else 'cls')
                self.display_banner()
            else:
                print("❌ اختيار غير صحيح، حاول مرة أخرى")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  تم إيقاف الإدخال بواسطة المستخدم")
        except Exception as e:
            print(f"❌ خطأ: {e}")
    
    def start_system(self):
        """بدء تشغيل النظام"""
        if self.radar_system.start():
            print("✅ بدأ تشغيل نظام الرادار")
            print("📡 جاري المسح الراداري...")
            print("📺 شاشة الرادار معروضة في الأعلى")
            print("\n⚠️  اضغط أي مفتاح للعودة إلى القائمة...")
            input()
        else:
            print("⚠️  النظام يعمل بالفعل")
    
    def stop_system(self):
        """إيقاف النظام"""
        if self.radar_system.stop():
            print("✅ تم إيقاف نظام الرادار")
        else:
            print("⚠️  النظام متوقف بالفعل")
    
    def show_system_info(self):
        """عرض معلومات النظام"""
        info = self.radar_system.get_system_info()
        
        print("\n" + "═" * 70)
        print("📊 معلومات النظام")
        print("═" * 70)
        
        for key, value in info.items():
            if key == 'statistics':
                continue
            
            # تحويل المفاتيح إلى عربية
            arabic_keys = {
                'runtime': 'مدة التشغيل',
                'radar_system': 'نظام الرادار',
                'status': 'الحالة',
                'mode': 'النمط',
                'targets_count': 'عدد الأهداف',
                'hostile_targets': 'الأهداف العدائية',
                'threat_level': 'مستوى التهديد'
            }
            
            display_key = arabic_keys.get(key, key)
            print(f"  {display_key:20}: {value}")
        
        print("═" * 70)
    
    def show_targets(self):
        """عرض الأهداف الحالية"""
        targets = self.radar_system.targets
        
        print("\n" + "═" * 70)
        print(f"🎯 الأهداف الحالية ({len(targets)})")
        print("═" * 70)
        
        if not targets:
            print("  لا توجد أهداف حالياً")
        else:
            for target_id, target in targets.items():
                range_km = np.linalg.norm(target.position) / 1000
                speed = np.linalg.norm(target.velocity)
                
                print(f"\n  🔹 الهدف: {target_id}")
                print(f"     النوع: {target.missile_type.value}")
                print(f"     الموقع: {range_km:.1f} كم")
                print(f"     السرعة: {speed:.0f} م/ث")
                print(f"     التهديد: {target.threat_level.value}")
                print(f"     الثقة: {target.confidence:.1f}%")
                print(f"     عدائي: {'نعم' if target.is_hostile else 'لا'}")
                print("     " + "─" * 40)
        
        print("═" * 70)
    
    def run_quick_simulation(self):
        """تشغيل محاكاة سريعة"""
        print("\n" + "═" * 70)
        print("🎮 بدء المحاكاة السريعة (60 ثانية)")
        print("═" * 70)
        
        # بدء النظام إذا لم يكن يعمل
        if not self.radar_system.is_active:
            self.radar_system.start()
            time.sleep(1)
        
        # إضافة أهداف عشوائية للمحاكاة
        for _ in range(5):
            self.radar_system.create_random_target()
        
        print("✅ تم إضافة 5 أهداف عشوائية")
        print("⏳ جاري المحاكاة... (اضغط Ctrl+C للإيقاف)")
        
        try:
            for i in range(60):  # 60 ثانية
                if not self.radar_system.is_active:
                    break
                
                time.sleep(1)
                
                # تحديث العرض كل 5 ثوان
                if (i + 1) % 5 == 0:
                    info = self.radar_system.get_system_info()
                    print(f"\n⏱️  الوقت: {i+1} ثانية")
                    print(f"🎯 الأهداف: {info['targets_count']}")
                    print(f"⚠️  التهديد: {info['threat_level']}")
                    
        except KeyboardInterrupt:
            print("\n\n⏹️  إيقاف المحاكاة...")
        
        # عرض النتائج
        print("\n" + "═" * 70)
        print("📊 نتائج المحاكاة")
        print("═" * 70)
        
        info = self.radar_system.get_system_info()
        stats = info['statistics']
        
        print(f"  مدة المحاكاة: {info['runtime']}")
        print(f"  عدد المسوح: {stats['total_scans']}")
        print(f"  الأهداف المكتشفة: {stats['targets_detected']}")
        print(f"  الصواريخ المحددة: {stats['missiles_identified']}")
        print(f"  معدل المسح: {stats['scan_rate']:.1f} هرتز")
        print("═" * 70)
        
        # إيقاف النظام بعد المحاكاة
        self.radar_system.stop()
    
    def show_radar_settings(self):
        """عرض إعدادات الرادار"""
        params = self.radar_system.params
        
        print("\n" + "═" * 70)
        print("⚙️  إعدادات الرادار")
        print("═" * 70)
        
        settings = [
            ("اسم النظام", params.name),
            ("التردد", f"{params.frequency/1e9:.1f} GHz"),
            ("القدرة", f"{params.power/1e3:.0f} كيلوواط"),
            ("القدرة القصوى", f"{params.peak_power/1e6:.1f} ميجاوات"),
            ("المدى الأقصى", f"{params.max_range/1000:.0f} كم"),
            ("دقة المدى", f"{params.range_resolution:.1f} م"),
            ("معدل النبضات", f"{params.prf:.0f} هرتز"),
            ("عرض النبضة", f"{params.pulse_width*1e6:.1f} ميكروثانية"),
            ("عرض الحزمة", f"{params.bandwidth/1e6:.1f} MHz"),
            ("زاوية الشعاع", f"{params.beam_width:.1f}°"),
            ("كسب الهوائي", f"{params.antenna_gain:.0f} ديسيبل"),
        ]
        
        for name, value in settings:
            print(f"  {name:20}: {value}")
        
        print("═" * 70)
    
    def show_statistics(self):
        """عرض إحصائيات النظام"""
        stats = self.radar_system.stats
        
        print("\n" + "═" * 70)
        print("📈 إحصائيات النظام")
        print("═" * 70)
        
        arabic_stats = {
            'total_scans': 'إجمالي المسوح',
            'targets_detected': 'الأهداف المكتشفة',
            'missiles_identified': 'الصواريخ المحددة',
            'false_alarms': 'الإنذارات الكاذبة',
            'scan_rate': 'معدل المسح (هرتز)'
        }
        
        for key, value in stats.items():
            if key == 'start_time':
                continue
            
            display_name = arabic_stats.get(key, key)
            if key == 'scan_rate':
                print(f"  {display_name:25}: {value:.1f}")
            else:
                print(f"  {display_name:25}: {value}")
        
        print("═" * 70)
    
    def save_system_data(self):
        """حفظ بيانات النظام"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"amds_data_{timestamp}.json"
            
            data = {
                'system_info': self.radar_system.get_system_info(),
                'targets': [],
                'detection_history': list(self.radar_system.detection_history)
            }
            
            # إضافة بيانات الأهداف
            for target_id, target in self.radar_system.targets.items():
                target_data = {
                    'id': target.id,
                    'position': target.position.tolist(),
                    'velocity': target.velocity.tolist(),
                    'missile_type': target.missile_type.value,
                    'threat_level': target.threat_level.value,
                    'confidence': target.confidence,
                    'is_hostile': target.is_hostile
                }
                data['targets'].append(target_data)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ تم حفظ بيانات النظام في: {filename}")
            
        except Exception as e:
            print(f"❌ خطأ في حفظ البيانات: {e}")
    
    def add_random_target(self):
        """إضافة هدف عشوائي"""
        target_id = self.radar_system.create_random_target()
        if target_id:
            print(f"✅ تم إضافة الهدف: {target_id}")
        else:
            print("❌ تعذر إضافة الهدف")
    
    def show_help(self):
        """عرض المساعدة"""
        print("\n" + "═" * 70)
        print("🆘 مساعدة نظام AMDS")
        print("═" * 70)
        print("  هذا نظام محاكاة لكشف الصواريخ باستخدام الرادارات المتطورة.")
        print("  تم تطويره للأغراض الأكاديمية والبحثية.")
        print("\n  الأوامر السريعة:")
        print("    S - بدء/إيقاف النظام")
        print("    A - إضافة هدف عشوائي")
        print("    C - مسح الشاشة")
        print("    Q - خروج")
        print("\n  المفاتيح:")
        print("    🟢 آمن       🟡 منخفض      🟠 متوسط")
        print("    🔴 عالي      💀 حرج        🚀 صاروخ")
        print("    ✈️  طائرة     🛰️  صاروخ بالستي")
        print("═" * 70)
    
    def exit_system(self):
        """خروج من النظام"""
        print("\n" + "═" * 70)
        print("🚪 تأكيد الخروج")
        print("═" * 70)
        
        confirm = input("هل تريد الخروج؟ (نعم/لا): ").strip().lower()
        
        if confirm in ['نعم', 'y', 'yes']:
            print("\n⏹️  جاري إيقاف النظام...")
            self.radar_system.stop()
            self.running = False
            print("✅ تم إيقاف النظام")
            print("👋 مع السلامة!")
            print("═" * 70)
    
    def run(self):
        """تشغيل الواجهة الرئيسية"""
        self.display_banner()
        
        while self.running:
            self.display_main_menu()
            self.handle_user_input()

# ============================================
# الجزء 6: التشغيل الرئيسي
# ============================================

def check_dependencies():
    """التحقق من تثبيت المكتبات المطلوبة"""
    required_libs = ['numpy']
    
    print("🔍 جاري التحقق من المكتبات المطلوبة...")
    
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"✅ {lib} مثبت")
        except ImportError:
            print(f"❌ {lib} غير مثبت")
            print(f"📦 جاري التثبيت التلقائي لـ {lib}...")
            
            try:
                import subprocess
                import sys
                
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                print(f"✅ تم تثبيت {lib} بنجاح")
                
                # إعادة تشغيل البرنامج
                print("🔁 جاري إعادة التشغيل...")
                os.execv(sys.executable, ['python'] + sys.argv)
                
            except:
                print(f"⚠️  تعذر تثبيت {lib} تلقائياً")
                print("يرجى تثبيته يدوياً:")
                print(f"  pip install {lib}")
                return False
    
    print("✅ جميع المكتبات المطلوبة مثبتة")
    return True

def main():
    """الدالة الرئيسية"""
    try:
        # التحقق من المكتبات
        if not check_dependencies():
            return
        
        # إنشاء وتشغيل الواجهة
        interface = AMDSInterface()
        interface.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  تم إيقاف البرنامج بواسطة المستخدم")
    except Exception as e:
        print(f"\n❌ خطأ غير متوقع: {e}")
        print("يرجى التأكد من تثبيت جميع المكتبات المطلوبة")
        print("يمكنك تثبيتها باستخدام:")
        print("  pip install numpy")
    finally:
        print("\n" + "═" * 70)
        print("شكراً لاستخدامك نظام كشف الصواريخ المتقدم (AMDS)")
        print("تم تطويره للأغراض الأكاديمية والبحثية")
        print("═" * 70)

# ============================================
# بدء التشغيل
# ============================================

if __name__ == "__main__":
    # تشغيل النظام
    main()
