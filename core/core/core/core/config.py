# utils/config.py - إعدادات النظام
import json

class SystemConfig:
    """إعدادات نظام الرادار"""
    
    def __init__(self, config_file=None):
        self.radar_settings = {
            'frequency': 10e9,
            'power': 1000,
            'range_max': 500000,
            'resolution': 10,
            'prf': 3000,
            'pulse_width': 100e-6
        }
        
        self.detection_settings = {
            'cfar_threshold': 1e-6,
            'min_snr': 10,
            'tracking_enabled': True
        }
        
        self.display_settings = {
            'refresh_rate': 10,
            'color_theme': 'dark',
            'show_grid': True
        }
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, filepath):
        """تحميل الإعدادات من ملف"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
                self.radar_settings.update(config.get('radar', {}))
                self.detection_settings.update(config.get('detection', {}))
                self.display_settings.update(config.get('display', {}))
        except:
            print("⚠️  استخدام الإعدادات الافتراضية")
    
    def save_config(self, filepath):
        """حفظ الإعدادات إلى ملف"""
        config = {
            'radar': self.radar_settings,
            'detection': self.detection_settings,
            'display': self.display_settings
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
