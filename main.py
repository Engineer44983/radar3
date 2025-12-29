#!/usr/bin/env python3
# main.py - النظام الرئيسي لكشف الصواريخ
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import logging
from datetime import datetime
from PyQt5.QtWidgets import QApplication
from core.radar_system import AdvancedRadarSystem
from gui.main_window import MainWindow
from utils.config import SystemConfig

# إعداد نظام التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AdvancedMissileDefenseSystem:
    """النظام الرئيسي للدفاع الجوي وكشف الصواريخ"""
    
    def __init__(self):
        logger.info("تهيئة نظام كشف الصواريخ المتقدم...")
        self.config = SystemConfig()
        self.radar_system = AdvancedRadarSystem(self.config)
        self.gui = None
        self.is_running = False
        
    def initialize_system(self):
        """تهيئة جميع مكونات النظام"""
        try:
            logger.info("جارٍ تهيئة النظام...")
            
            # تهيئة نظام الرادار
            if self.radar_system.initialize():
                logger.info("✅ تم تهيئة نظام الرادار بنجاح")
            else:
                logger.error("❌ فشل في تهيئة نظام الرادار")
                return False
            
            # تحميل نماذج الذكاء الاصطناعي
            self.load_ai_models()
            
            logger.info("✅ تم تهيئة النظام بالكامل بنجاح")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في تهيئة النظام: {str(e)}")
            return False
    
    def load_ai_models(self):
        """تحميل نماذج التعلم الآلي"""
        try:
            from ai.missile_classifier import MissileClassifier
            from ai.trajectory_predictor import TrajectoryPredictor
            
            self.classifier = MissileClassifier()
            self.trajectory_predictor = TrajectoryPredictor()
            
            # تحميل النماذج المدربة
            if os.path.exists("ai_models/classifier_model.pkl"):
                self.classifier.load_model("ai_models/classifier_model.pkl")
            if os.path.exists("ai_models/trajectory_model.pkl"):
                self.trajectory_predictor.load_model("ai_models/trajectory_model.pkl")
                
            logger.info("✅ تم تحميل نماذج الذكاء الاصطناعي")
            
        except Exception as e:
            logger.warning(f"تعذر تحميل نماذج الذكاء الاصطناعي: {str(e)}")
    
    def start_system(self):
        """بدء تشغيل النظام"""
        if not self.is_running:
            logger.info("بدء تشغيل نظام الكشف...")
            self.is_running = True
            self.radar_system.start()
            
            # بدء دورة المعالجة الرئيسية
            self.main_cycle()
    
    def main_cycle(self):
        """الدورة الرئيسية للمعالجة"""
        import time
        cycle_count = 0
        
        while self.is_running:
            try:
                cycle_count += 1
                
                # جمع البيانات من الرادار
                radar_data = self.radar_system.get_current_data()
                
                if radar_data and len(radar_data['targets']) > 0:
                    # معالجة الأهداف المكتشفة
                    processed_targets = self.process_targets(radar_data['targets'])
                    
                    # تحليل التهديد
                    threat_assessment = self.assess_threat_level(processed_targets)
                    
                    # تحديث الواجهة الرسومية
                    if self.gui:
                        self.gui.update_display(processed_targets, threat_assessment)
                    
                    # تسجيل النتائج
                    self.log_detection(processed_targets, cycle_count)
                
                time.sleep(0.1)  # 10 هرتز
                
            except KeyboardInterrupt:
                logger.info("إيقاف النظام بواسطة المستخدم")
                self.stop_system()
                break
            except Exception as e:
                logger.error(f"خطأ في الدورة الرئيسية: {str(e)}")
                time.sleep(1)
    
    def process_targets(self, targets):
        """معالجة وتصنيف الأهداف المكتشفة"""
        processed = []
        
        for target in targets:
            # تطبيق مرشح كالمان
            filtered_target = self.apply_kalman_filter(target)
            
            # تصنيف الهدف باستخدام الذكاء الاصطناعي
            classification = self.classify_target(filtered_target)
            
            # توقع المسار
            trajectory = self.predict_trajectory(filtered_target)
            
            processed.append({
                'id': target.get('id', hash(str(target))),
                'position': filtered_target['position'],
                'velocity': filtered_target['velocity'],
                'classification': classification,
                'trajectory': trajectory,
                'threat_level': self.calculate_threat_level(filtered_target, classification),
                'timestamp': datetime.now(),
                'raw_data': target
            })
        
        return processed
    
    def apply_kalman_filter(self, target):
        """تطبيق مرشح كالمان لتنقية البيانات"""
        from filters.kalman_filter import KalmanFilter3D
        
        kf = KalmanFilter3D()
        
        # قياسات الموقع والسرعة
        measurement = np.array([
            target['x'], target['y'], target['z'],
            target.get('vx', 0), target.get('vy', 0), target.get('vz', 0)
        ])
        
        # تطبيق المرشح
        filtered_state = kf.update(measurement)
        
        return {
            'position': filtered_state[:3],
            'velocity': filtered_state[3:6],
            'covariance': kf.get_covariance()
        }
    
    def classify_target(self, target):
        """تصنيف الهدف باستخدام التعلم الآلي"""
        try:
            features = self.extract_features(target)
            classification = self.classifier.predict(features)
            return classification
        except:
            # تصنيف بدائي في حالة فشل النموذج
            speed = np.linalg.norm(target['velocity'])
            if speed > 1000:  # م/ث
                return "صاروخ باليستي"
            elif speed > 300:
                return "صاروخ كروز"
            else:
                return "طائرة"
    
    def extract_features(self, target):
        """استخراج مميزات الهدف للتصنيف"""
        pos = target['position']
        vel = target['velocity']
        
        features = [
            np.linalg.norm(vel),  # السرعة
            np.linalg.norm(pos),  # المسافة
            vel[2] if len(vel) > 2 else 0,  # السرعة العمودية
            np.std(pos),  # تغير الموقع
            np.mean(np.abs(vel))  # متوسط السرعة
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_trajectory(self, target):
        """توقع مسار الهدف"""
        try:
            trajectory = self.trajectory_predictor.predict(
                target['position'],
                target['velocity']
            )
            return trajectory
        except:
            # توقع خطي بسيط
            steps = 10
            pos = target['position']
            vel = target['velocity']
            
            trajectory = []
            for i in range(steps):
                time_step = i * 0.5  # 0.5 ثانية بين النقاط
                future_pos = pos + vel * time_step
                trajectory.append(future_pos)
            
            return np.array(trajectory)
    
    def calculate_threat_level(self, target, classification):
        """حساب مستوى التهديد"""
        threat_score = 0
        
        # عامل السرعة
        speed = np.linalg.norm(target['velocity'])
        if speed > 2000:
            threat_score += 100
        elif speed > 1000:
            threat_score += 80
        elif speed > 500:
            threat_score += 50
        
        # عامل الارتفاع
        altitude = target['position'][2] if len(target['position']) > 2 else 0
        if 0 < altitude < 1000:  # منخفض
            threat_score += 70
        elif 1000 <= altitude < 10000:  # متوسط
            threat_score += 40
        else:  # عالي
            threat_score += 20
        
        # عامل التصنيف
        if classification == "صاروخ باليستي":
            threat_score += 90
        elif classification == "صاروخ كروز":
            threat_score += 70
        elif classification == "طائرة":
            threat_score += 30
        
        # حساب مستوى التهديد النهائي
        if threat_score >= 150:
            return "حرج"
        elif threat_score >= 100:
            return "عالي"
        elif threat_score >= 50:
            return "متوسط"
        else:
            return "منخفض"
    
    def assess_threat_level(self, targets):
        """تقييم مستوى التهديد العام"""
        if not targets:
            return {"level": "آمن", "color": "green"}
        
        max_threat = max(t.get('threat_level_score', 0) for t in targets)
        
        if max_threat >= 150:
            return {"level": "تهديد حرج", "color": "red", "alarm": True}
        elif max_threat >= 100:
            return {"level": "تهديد عالي", "color": "orange", "alarm": True}
        elif max_threat >= 50:
            return {"level": "تهديد متوسط", "color": "yellow", "alarm": False}
        else:
            return {"level": "وضع آمن", "color": "green", "alarm": False}
    
    def log_detection(self, targets, cycle_id):
        """تسجيل عمليات الكشف"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'cycle': cycle_id,
            'targets_count': len(targets),
            'threats': [
                {
                    'id': t['id'],
                    'position': t['position'].tolist(),
                    'classification': t['classification'],
                    'threat_level': t['threat_level']
                }
                for t in targets
            ]
        }
        
        # حفظ في ملف JSON
        import json
        with open(f'logs/detections_{datetime.now().strftime("%Y%m%d")}.json', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def stop_system(self):
        """إيقاف النظام"""
        logger.info("إيقاف نظام الكشف...")
        self.is_running = False
        if self.radar_system:
            self.radar_system.stop()
    
    def run_gui(self):
        """تشغيل الواجهة الرسومية"""
        app = QApplication(sys.argv)
        self.gui = MainWindow(self)
        self.gui.show()
        sys.exit(app.exec_())

def main():
    """الدالة الرئيسية"""
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║    نظام كشف الصواريخ المتقدم (AMDS) v2.0            ║
    ║    Advanced Missile Detection System                ║
    ║    تم التطوير للأغراض الأكاديمية والبحثية           ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # التحقق من صلاحية المستخدم
    if os.geteuid() == 0:
        print("⚠️  تحذير: لا تشغل النظام كجذر (root)")
        choice = input("هل تريد المتابعة؟ (نعم/لا): ")
        if choice.lower() not in ['نعم', 'y', 'yes']:
            sys.exit(0)
    
    # إنشاء النظام
    amds = AdvancedMissileDefenseSystem()
    
    # التهيئة
    if amds.initialize_system():
        print("✅ النظام جاهز للتشغيل")
        
        # اختيار الوضع
        print("\nاختر وضع التشغيل:")
        print("1. الوضع الرسومي (GUI)")
        print("2. الوضع الطرفي (Terminal)")
        print("3. وضع المحاكاة (Simulation Only)")
        
        try:
            choice = int(input("اختيارك (1-3): "))
            
            if choice == 1:
                amds.run_gui()
            elif choice == 2:
                amds.start_system()
            elif choice == 3:
                from simulations.full_simulation import run_comprehensive_simulation
                run_comprehensive_simulation()
            else:
                print("❌ اختيار غير صحيح")
                
        except KeyboardInterrupt:
            print("\nتم إيقاف النظام")
        except Exception as e:
            print(f"❌ خطأ: {str(e)}")
    else:
        print("❌ فشل في تهيئة النظام")
        sys.exit(1)

if __name__ == "__main__":
    main()
