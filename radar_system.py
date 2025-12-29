# core/radar_system.py - نظام الرادار المتقدم
import numpy as np
import threading
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class RadarMode(Enum):
    """أنماط عمل الرادار"""
    SEARCH = "search"
    TRACK = "track"
    TRACK_WHILE_SCAN = "track_while_scan"
    ILLUMINATION = "illumination"

class AdvancedRadarSystem:
    """نظام رادار متطور لكشف الصواريخ"""
    
    def __init__(self, config):
        self.config = config
        self.mode = RadarMode.SEARCH
        self.frequency = 10e9  # 10 GHz
        self.power = 1000  # كيلوواط
        self.range_max = 500000  # 500 كم
        self.resolution = 10  # أمتار
        self.beam_width = 1.5  # درجات
        
        self.targets = []
        self.clutter = []
        self.is_active = False
        self.thread = None
        
        # مصفوفات الهوائي الطورية
        self.phase_array = self.initialize_phase_array()
        
        # معلمات معالجة الإشارات
        self.pulse_width = 100e-6  # ثانية
        self.pulse_repetition_frequency = 3000  # هرتز
        self.bandwidth = 10e6  # 10 MHz
        
        logger.info("تم إنشاء نظام الرادار المتقدم")
    
    def initialize_phase_array(self):
        """تهيئة مصفوفة الهوائي الطورية"""
        array_size = (32, 32)  # 1024 عنصر
        return np.zeros(array_size)
    
    def initialize(self):
        """تهيئة النظام"""
        try:
            # محاكاة تهيئة المكونات المادية
            self.calibrate_system()
            self.load_clutter_models()
            self.set_mode(RadarMode.SEARCH)
            
            logger.info("تهيئة نظام الرادار...")
            return True
        except Exception as e:
            logger.error(f"خطأ في تهيئة الرادار: {str(e)}")
            return False
    
    def calibrate_system(self):
        """معايرة النظام"""
        logger.info("جارٍ معايرة نظام الرادار...")
        time.sleep(1)  # محاكاة وقت المعايرة
        logger.info("✅ تمت المعايرة بنجاح")
    
    def load_clutter_models(self):
        """تحميل نماذج الفوضى الأرضية والجوية"""
        # نماذج بسيطة للفوضى
        self.clutter = [
            {'type': 'ground', 'range': 5000, 'rcs': 1000},
            {'type': 'weather', 'range': 15000, 'rcs': 500},
            {'type': 'birds', 'range': 1000, 'rcs': 0.01},
        ]
    
    def set_mode(self, mode):
        """تغيير نمط عمل الرادار"""
        self.mode = mode
        logger.info(f"تم تغيير نمط الرادار إلى: {mode.value}")
    
    def start(self):
        """بدء تشغيل الرادار"""
        if not self.is_active:
            self.is_active = True
            self.thread = threading.Thread(target=self.radar_cycle, daemon=True)
            self.thread.start()
            logger.info("بدأ تشغيل الرادار")
    
    def stop(self):
        """إيقاف الرادار"""
        self.is_active = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("تم إيقاف الرادار")
    
    def radar_cycle(self):
        """دورة عمل الرادار"""
        cycle_count = 0
        
        while self.is_active:
            try:
                cycle_count += 1
                
                # توليد نبضة رادارية
                pulse = self.generate_pulse()
                
                # محاكاة انتشار النبضة
                transmitted_signal = self.transmit_pulse(pulse)
                
                # محاكاة الأهداف
                self.generate_simulated_targets()
                
                # محاكاة الإشارات المرتدة
                received_signals = self.simulate_reception(transmitted_signal)
                
                # معالجة الإشارات
                processed_data = self.process_signals(received_signals)
                
                # كشف الأهداف
                detected_targets = self.detect_targets(processed_data)
                
                # تحديث قائمة الأهداف
                self.update_targets(detected_targets)
                
                # تسجيل البيانات
                if cycle_count % 10 == 0:
                    logger.debug(f"دورة الرادار {cycle_count}: تم كشف {len(self.targets)} هدف")
                
                time.sleep(0.05)  # 20 هرتز
                
            except Exception as e:
                logger.error(f"خطأ في دورة الرادار: {str(e)}")
                time.sleep(1)
    
    def generate_pulse(self):
        """توليد نبضة رادارية"""
        # نبضة خطية التردد (LFM)
        t = np.linspace(0, self.pulse_width, int(self.pulse_width * 100e6))
        f0 = self.frequency
        bandwidth = self.bandwidth
        
        # تعديل التردد
        chirp = bandwidth / self.pulse_width
        phase = 2 * np.pi * (f0 * t + 0.5 * chirp * t**2)
        
        pulse = np.exp(1j * phase)
        return pulse
    
    def transmit_pulse(self, pulse):
        """محاكاة إرسال النبضة"""
        # إضافة خصائص الهوائي
        antenna_gain = self.calculate_antenna_gain()
        transmitted_power = self.power * antenna_gain
        
        # حساب خسائر الانتشار
        range_km = self.range_max / 1000
        propagation_loss = 32.4 + 20 * np.log10(self.frequency / 1e6) + 20 * np.log10(range_km)
        
        # إشارة مرسلة
        transmitted_signal = {
            'signal': pulse * np.sqrt(transmitted_power),
            'power': transmitted_power,
            'loss': propagation_loss,
            'timestamp': time.time()
        }
        
        return transmitted_signal
    
    def calculate_antenna_gain(self):
        """حساب كسب الهوائي"""
        # صيغة تقريبية لكسب الهوائي الطوري
        wavelength = 3e8 / self.frequency
        aperture_area = (32 * wavelength/2) * (32 * wavelength/2)
        gain = 4 * np.pi * aperture_area / wavelength**2
        return 10 * np.log10(gain)
    
    def generate_simulated_targets(self):
        """توليد أهداف محاكاة"""
        import random
        
        target_types = [
            {'name': 'صاروخ باليستي', 'speed': 2000, 'rcs': 0.1, 'altitude': 50000},
            {'name': 'صاروخ كروز', 'speed': 300, 'rcs': 0.5, 'altitude': 1000},
            {'name': 'طائرة مقاتلة', 'speed': 500, 'rcs': 5, 'altitude': 10000},
            {'name': 'طائرة تجارية', 'speed': 250, 'rcs': 100, 'altitude': 12000},
        ]
        
        num_targets = random.randint(0, 5)
        new_targets = []
        
        for _ in range(num_targets):
            target_type = random.choice(target_types)
            
            # توليد موقع عشوائي
            range_km = random.uniform(10, 400)
            azimuth = random.uniform(0, 360)
            elevation = random.uniform(0, 45)
            
            # تحويل إلى إحداثيات ديكارتية
            x = range_km * 1000 * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))
            y = range_km * 1000 * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
            z = range_km * 1000 * np.sin(np.radians(elevation))
            
            # توليد سرعة
            speed = random.uniform(target_type['speed'] * 0.7, target_type['speed'] * 1.3)
            heading = random.uniform(0, 360)
            climb_angle = random.uniform(-10, 10)
            
            vx = speed * np.cos(np.radians(climb_angle)) * np.cos(np.radians(heading))
            vy = speed * np.cos(np.radians(climb_angle)) * np.sin(np.radians(heading))
            vz = speed * np.sin(np.radians(climb_angle))
            
            target = {
                'id': random.randint(1000, 9999),
                'type': target_type['name'],
                'position': np.array([x, y, z]),
                'velocity': np.array([vx, vy, vz]),
                'rcs': target_type['rcs'],
                'snr': random.uniform(10, 30),
                'detection_time': time.time(),
                'update_time': time.time()
            }
            
            new_targets.append(target)
        
        # تحديث الأهداف الحالية
        self.update_target_positions()
        
        # إضافة أهداف جديدة
        for target in new_targets:
            # تجنب التكرار
            if not any(t['id'] == target['id'] for t in self.targets):
                self.targets.append(target)
    
    def update_target_positions(self):
        """تحديث مواقع الأهداف بناءً على سرعاتها"""
        current_time = time.time()
        
        for target in self.targets:
            dt = current_time - target['update_time']
            target['position'] += target['velocity'] * dt
            target['update_time'] = current_time
            
            # حذف الأهداف خارج النطاق
            range_distance = np.linalg.norm(target['position'])
            if range_distance > self.range_max:
                self.targets.remove(target)
    
    def simulate_reception(self, transmitted_signal):
        """محاكاة استقبال الإشارات"""
        received_signals = []
        
        # إضافة إشارات الأهداف
        for target in self.targets:
            # حساب التأخير الزمني
            range_distance = np.linalg.norm(target['position'])
            time_delay = 2 * range_distance / 3e8
            
            # حساب خسائر المسار
            wavelength = 3e8 / self.frequency
            rcs = target['rcs']
            path_loss = (wavelength**2 * rcs) / ((4 * np.pi)**3 * range_distance**4)
            
            # تأثير دوبلر
            radial_velocity = self.calculate_radial_velocity(target)
            doppler_shift = 2 * radial_velocity / wavelength
            
            # إشارة مستلمة من الهدف
            target_signal = {
                'signal': transmitted_signal['signal'] * np.sqrt(path_loss),
                'delay': time_delay,
                'doppler': doppler_shift,
                'snr': target['snr'],
                'target_id': target['id']
            }
            
            received_signals.append(target_signal)
        
        # إضافة ضوضاء
        noise_signal = self.add_noise(transmitted_signal['signal'].shape)
        received_signals.append({
            'signal': noise_signal,
            'delay': 0,
            'doppler': 0,
            'snr': 0,
            'target_id': 'noise'
        })
        
        return received_signals
    
    def calculate_radial_velocity(self, target):
        """حساب السرعة الشعاعية"""
        # متجه الوحدة في اتجاه الرادار
        radar_direction = -target['position'] / np.linalg.norm(target['position'])
        
        # السرعة الشعاعية (المركبة في اتجاه الرادار)
        radial_velocity = np.dot(target['velocity'], radar_direction)
        
        return radial_velocity
    
    def add_noise(self, shape):
        """إضافة ضوضاء غوسية بيضاء"""
        noise_power = 0.1  # قدرة الضوضاء
        noise_real = np.random.normal(0, np.sqrt(noise_power/2), shape)
        noise_imag = np.random.normal(0, np.sqrt(noise_power/2), shape)
        return noise_real + 1j * noise_imag
    
    def process_signals(self, received_signals):
        """معالجة الإشارات المستلمة"""
        processed_data = {
            'range_profile': None,
            'doppler_profile': None,
            'angle_of_arrival': None,
            'timestamp': time.time()
        }
        
        # تجميع جميع الإشارات
        combined_signal = np.zeros_like(received_signals[0]['signal'], dtype=complex)
        
        for rx_signal in received_signals:
            signal = rx_signal['signal']
            delay_samples = int(rx_signal['delay'] * 100e6)  # تحويل إلى عينات
            
            # تطبيق التأخير
            if delay_samples < len(signal):
                shifted_signal = np.roll(signal, delay_samples)
                shifted_signal[:delay_samples] = 0
                
                # تطبيق تأثير دوبلر
                t = np.linspace(0, len(signal)/100e6, len(signal))
                doppler_phase = 2 * np.pi * rx_signal['doppler'] * t
                doppler_signal = shifted_signal * np.exp(1j * doppler_phase)
                
                combined_signal += doppler_signal
        
        # تحليل المدى (FFT)
        range_fft = np.fft.fft(combined_signal)
        processed_data['range_profile'] = np.abs(range_fft)
        
        # تحليل دوبلر
        # في النظام الحقيقي، يتم جمع عدة نبضات
        num_pulses = 32
        doppler_matrix = np.zeros((num_pulses, len(combined_signal)), dtype=complex)
        
        for i in range(num_pulses):
            # محاكاة نبضات متتالية
            doppler_matrix[i, :] = combined_signal * np.exp(1j * 2 * np.pi * i / num_pulses)
        
        doppler_fft = np.fft.fft(doppler_matrix, axis=0)
        processed_data['doppler_profile'] = np.abs(doppler_fft)
        
        # حساب زاوية الوصول (للرادار الطوري)
        processed_data['angle_of_arrival'] = self.calculate_aoa(combined_signal)
        
        return processed_data
    
    def calculate_aoa(self, signal):
        """حساب زاوية الوصول باستخدام المصفوفة الطورية"""
        # طريقة MUSIC بسيطة (في النظام الحقيقي ستكون أكثر تعقيداً)
        angles = np.linspace(-45, 45, 181)
        spectral_power = np.zeros_like(angles)
        
        for i, angle in enumerate(angles):
            # متجه التأخير المتوقع
            steering_vector = self.calculate_steering_vector(np.radians(angle))
            
            # طيف الطاقة
            spectral_power[i] = np.abs(np.dot(steering_vector.conj().T, np.mean(signal)))**2
        
        # إيجاد الذروة
        peak_idx = np.argmax(spectral_power)
        return angles[peak_idx]
    
    def calculate_steering_vector(self, angle):
        """حساب متجه التوجيه للهوائي الطوري"""
        num_elements = 32
        element_spacing = 0.5  # نصف الطول الموجي
        
        steering_vector = np.exp(1j * 2 * np.pi * element_spacing * 
                                np.arange(num_elements) * np.sin(angle))
        return steering_vector
    
    def detect_targets(self, processed_data):
        """كشف الأهداف من البيانات المعالجة"""
        detected_targets = []
        
        # تطبيق خوارزمية CFAR
        range_profile = processed_data['range_profile']
        threshold = self.cfar_detection(range_profile)
        
        # إيجاد الذروات فوق العتبة
        peaks = self.find_peaks(range_profile, threshold)
        
        for peak in peaks:
            # استخراج معلومات الهدف
            range_idx, doppler_idx = peak
            
            # تحويل الفهرس إلى قيم فيزيائية
            range_distance = (range_idx / len(range_profile)) * self.range_max
            doppler_freq = (doppler_idx / 32) * self.pulse_repetition_frequency
            
            # سرعة دوبلر
            wavelength = 3e8 / self.frequency
            radial_velocity = doppler_freq * wavelength / 2
            
            # البحث عن هدف مطابق في قائمة الأهداف المحاكاة
            matched_target = self.match_simulated_target(range_distance, radial_velocity)
            
            if matched_target:
                detected_targets.append(matched_target)
            else:
                # إنشاء هدف جديد
                target = {
                    'id': int(time.time() * 1000) % 10000,
                    'x': range_distance * np.cos(processed_data['angle_of_arrival']),
                    'y': range_distance * np.sin(processed_data['angle_of_arrival']),
                    'z': range_distance * 0.1,  # تقدير الارتفاع
                    'vx': radial_velocity * np.cos(processed_data['angle_of_arrival']),
                    'vy': radial_velocity * np.sin(processed_data['angle_of_arrival']),
                    'vz': 0,
                    'snr': 20,
                    'detection_time': time.time()
                }
                detected_targets.append(target)
        
        return detected_targets
    
    def cfar_detection(self, signal, pfa=1e-6):
        """خوارزمية CFAR للكشف التكيفي"""
        # CA-CFAR (Cell Averaging CFAR)
        guard_cells = 2
        reference_cells = 10
        
        thresholded_signal = np.zeros_like(signal)
        
        for i in range(len(signal)):
            # تجنب الحواف
            if i < reference_cells + guard_cells or i >= len(signal) - reference_cells - guard_cells:
                continue
            
            # الخلايا المرجعية
            reference_window = np.concatenate([
                signal[i - reference_cells - guard_cells:i - guard_cells],
                signal[i + guard_cells + 1:i + guard_cells + reference_cells + 1]
            ])
            
            # حساب العتبة
            noise_level = np.mean(reference_window)
            threshold = noise_level * (-np.log(pfa))
            
            # الكشف
            if signal[i] > threshold:
                thresholded_signal[i] = signal[i]
        
        return thresholded_signal
    
    def find_peaks(self, signal, thresholded_signal, min_distance=5):
        """إيجاد الذروات في الإشارة"""
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(
            thresholded_signal,
            height=np.max(thresholded_signal) * 0.1,
            distance=min_distance
        )
        
        # زوج من مؤشرات المدى ودوبلر
        peak_pairs = [(peak, peak % 32) for peak in peaks]
        
        return peak_pairs
    
    def match_simulated_target(self, range_distance, radial_velocity):
        """مطابقة الهدف مع الأهداف المحاكاة"""
        for target in self.targets:
            target_range = np.linalg.norm(target['position'])
            target_velocity = self.calculate_radial_velocity(target)
            
            # قبول هامش خطأ
            range_error = abs(target_range - range_distance) / range_distance
            velocity_error = abs(target_velocity - radial_velocity) / abs(radial_velocity) if radial_velocity != 0 else 0
            
            if range_error < 0.1 and velocity_error < 0.2:
                return {
                    'id': target['id'],
                    'x': target['position'][0],
                    'y': target['position'][1],
                    'z': target['position'][2],
