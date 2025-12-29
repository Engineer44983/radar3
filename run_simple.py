#!/usr/bin/env python3
# run_simple.py - نسخة مبسطة للتشغيل المباشر

import numpy as np
import time
import random

class SimpleRadarSystem:
    """نظام رادار مبسط يعمل بدون موديولات خارجية"""
    
    def __init__(self):
        print("""
╔══════════════════════════════════════════════════════╗
║    نظام محاكاة الرادار المتقدم (AMDS Lite)          ║
║    تم التطوير للأغراض الأكاديمية                    ║
╚══════════════════════════════════════════════════════╝
        """)
        
        self.radar_params = {
            'name': 'AN/SPY-1D (محاكاة)',
            'range': 500,  # كم
            'frequency': 'S-band',
            'power': '4-6 MW',
            'targets_detected': 0
        }
        
        self.simulation_running = True
    
    def simulate_scan(self):
        """محاكاة عملية مسح الرادار"""
        print("\\n🔍 جاري مسح المجال الجوي...")
        time.sleep(1)
        
        # محاكاة اكتشاف أهداف
        num_targets = random.randint(0, 4)
        targets = []
        
        for i in range(num_targets):
            target_type = random.choice([
                ("صاروخ باليستي", 2500, "🛰️"),
                ("صاروخ كروز", 300, "🚀"),
                ("طائرة مقاتلة", 600, "✈️"),
                ("طائرة بدون طيار", 150, "🛸")
            ])
            
            target = {
                'id': f"TGT-{random.randint(1000, 9999)}",
                'type': target_type[0],
                'icon': target_type[2],
                'range': random.uniform(50, 450),
                'bearing': random.uniform(0, 359.9),
                'speed': random.uniform(target_type[1] * 0.8, target_type[1] * 1.2),
                'altitude': random.uniform(500, 30000),
                'threat': random.choice(['منخفض', 'متوسط', 'عالي', 'حرج'])
            }
            targets.append(target)
        
        return targets
    
    def display_radar_screen(self, targets):
        """عرض شاشة الرادار"""
        print("\\n" + "=" * 60)
        print("📡 شاشة الرادار - النطاق: 500 كم")
        print("=" * 60)
        
        if not targets:
            print("\\n       ⭕ لا توجد أهداف في النطاق")
            print("\\n       منطقة آمنة")
        else:
            print(f"\\n🎯 تم اكتشاف {len(targets)} هدف:")
            print("-" * 60)
            
            for target in targets:
                threat_color = {
                    'منخفض': '🟢',
                    'متوسط': '🟡',
                    'عالي': '🟠',
                    'حرج': '🔴'
                }.get(target['threat'], '⚪')
                
                print(f"{target['icon']} {target['id']}: {target['type']}")
                print(f"   المدى: {target['range']:.1f} كم | الاتجاه: {target['bearing']:.1f}°")
                print(f"   السرعة: {target['speed']:.0f} م/ث | الارتفاع: {target['altitude']:.0f} م")
                print(f"   مستوى التهديد: {threat_color} {target['threat']}")
                print("-" * 40)
        
        print("\\n" + "=" * 60)
        print("مفاتيح الألوان: 🟢 منخفض 🟡 متوسط 🟠 عالي 🔴 حرج")
        print("=" * 60)
    
    def calculate_threat_assessment(self, targets):
        """حساب تقييم التهديد"""
        if not targets:
            return "✅ الوضع: آمن"
        
        threat_levels = [t['threat'] for t in targets]
        
        if 'حرج' in threat_levels:
            return "🚨 الوضع: تأهب قصوى - تهديد حرج!"
        elif 'عالي' in threat_levels:
            return "⚠️  الوضع: تأهب عالي"
        elif 'متوسط' in threat_levels:
            return "🔶 الوضع: تأهب متوسط"
        else:
            return "✅ الوضع: تحت السيطرة"
    
    def run_simulation(self):
        """تشغيل المحاكاة"""
        print(f"\\n📡 نظام الرادار: {self.radar_params['name']}")
        print(f"📊 المدى الأقصى: {self.radar_params['range']} كم")
        print(f"📶 التردد: {self.radar_params['frequency']}")
        print("\\n" + "─" * 50)
        
        cycle = 1
        try:
            while self.simulation_running:
                print(f"\\n🌀 دورة المسح #{cycle}")
                print("─" * 30)
                
                # محاكاة المسح
                targets = self.simulate_scan()
                self.radar_params['targets_detected'] = len(targets)
                
                # عرض النتائج
                self.display_radar_screen(targets)
                
                # تقييم التهديد
                threat_assessment = self.calculate_threat_assessment(targets)
                print(f"\\n{threat_assessment}")
                
                # إحصائيات
                print(f"\\n📈 الإحصائيات:")
                print(f"   - عدد الدورات: {cycle}")
                print(f"   - إجمالي الأهداف المكتشفة: {self.radar_params['targets_detected']}")
                
                cycle += 1
                
                # انتظر للمتابعة
                print("\\n" + "─" * 50)
                try:
                    cont = input("أدخل 'q' للإيقاف أو 'Enter' للمتابعة: ")
                    if cont.lower() == 'q':
                        break
                except KeyboardInterrupt:
                    break
                    
        except Exception as e:
            print(f"\\n❌ خطأ: {e}")
        
        print("\\n✅ تم إنهاء المحاكاة")
        print(f"📊 النتائج النهائية: {cycle-1} دورات مسح")

def main():
    """الدالة الرئيسية"""
    radar = SimpleRadarSystem()
    radar.run_simulation()

if __name__ == "__main__":
    main()
