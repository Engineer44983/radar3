#!/usr/bin/env python3
# setup_project.py - إنشاء هيكل المشروع الكامل

import os
import sys

# هيكل المجلدات المطلوب
project_structure = {
    'core': [
        '__init__.py',
        'radar_system.py',
        'signal_processor.py',
        'threat_analyzer.py'
    ],
    'gui': [
        '__init__.py',
        'main_window.py',
        'radar_display.py',
        'control_panel.py'
    ],
    'ai': [
        '__init__.py',
        'missile_classifier.py',
        'trajectory_predictor.py'
    ],
    'filters': [
        '__init__.py',
        'kalman_filter.py',
        'cfar_detector.py'
    ],
    'simulations': [
        '__init__.py',
        'full_simulation.py',
        'target_generator.py'
    ],
    'utils': [
        '__init__.py',
        'config.py',
        'logger.py',
        'helpers.py'
    ],
    'data': {
        'training': [],
        'simulations': []
    },
    'logs': [],
    'ai_models': []
}

def create_structure(base_path='.'):
    """إنشاء هيكل المجلدات والملفات"""
    
    for folder, contents in project_structure.items():
        folder_path = os.path.join(base_path, folder)
        
        # إنشاء المجلد إذا لم يكن موجوداً
        os.makedirs(folder_path, exist_ok=True)
        print(f"✓ تم إنشاء مجلد: {folder_path}")
        
        # إذا كان المحتوى قائمة ملفات
        if isinstance(contents, list):
            for file in contents:
                file_path = os.path.join(folder_path, file)
                
                # إنشاء ملف __init__.py فارغ
                if file == '__init__.py':
                    with open(file_path, 'w') as f:
                        f.write('# Package initialization\n')
                    print(f"  ✓ تم إنشاء: {file}")
                
                # إنشاء ملفات أخرى بقوالب أساسية
                elif file.endswith('.py'):
                    with open(file_path, 'w') as f:
                        f.write(f'# {file} - ملف تلقائي الإنشاء\n\n')
                        f.write('"""ملف جزء من نظام AMDS"""\n\n')
                    print(f"  ✓ تم إنشاء: {file}")

def create_main_files():
    """إنشاء الملفات الرئيسية"""
    
    # main.py (مبسط)
    with open('main.py', 'w') as f:
        f.write('''#!/usr/bin/env python3
# main.py - النظام الرئيسي لكشف الصواريخ

import sys
import os

# إضافة مسار المجلدات إلى نظام Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'gui'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def main():
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║    نظام كشف الصواريخ المتقدم (AMDS) v2.0            ║
    ║    Advanced Missile Detection System                ║
    ║    تم التطوير للأغراض الأكاديمية والبحثية           ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    print("🚀 النظام قيد التشغيل...")
    
    try:
        # استيراد الموديولات بعد إضافة المسارات
        from core.radar_system import AdvancedRadarSystem
        print("✅ تم تحميل نظام الرادار بنجاح")
        
        # إنشاء مثيل النظام
        radar = AdvancedRadarSystem()
        print("✅ تم تهيئة النظام")
        
        # بدء تشغيل النظام
        print("\\n🔍 بدء المسح الراداري...")
        radar.start_simulation()
        
    except ImportError as e:
        print(f"❌ خطأ في استيراد الموديولات: {e}")
        print("تأكد من وجود جميع الملفات المطلوبة")
    except Exception as e:
        print(f"❌ خطأ غير متوقع: {e}")

if __name__ == "__main__":
    main()
''')
    
    print("✓ تم إنشاء main.py")
    
    # requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write('''numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.3
PyQt5>=5.15.6
pyqtgraph>=0.12.3
scikit-learn>=0.24.2
pandas>=1.3.0
numba>=0.53.1
pygame>=2.0.1
''')
    
    print("✓ تم إنشاء requirements.txt")

if __name__ == "__main__":
    print("🚀 جاري إنشاء هيكل مشروع AMDS...")
    print("=" * 50)
    
    create_structure()
    create_main_files()
    
    print("=" * 50)
    print("✅ تم إنشاء هيكل المشروع بنجاح!")
    print("\\nلتشغيل النظام:")
    print("1. قم بتثبيت المتطلبات: pip install -r requirements.txt")
    print("2. شغل النظام: python main.py")
