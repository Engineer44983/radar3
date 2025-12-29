# adsb_receiver.py
import json
import socket
import threading
import time
from datetime import datetime
import mysql.connector
import folium
from flask import Flask, render_template_string

class ADSBReceiver:
    def __init__(self, host='localhost', port=30003):
        """
        نظام استقبال بيانات ADS-B من dump1090
        """
        self.host = host
        self.port = port
        self.aircraft_data = {}
        self.running = False
        self.db_connection = self.init_database()
        
    def init_database(self):
        """تهيئة قاعدة البيانات"""
        try:
            conn = mysql.connector.connect(
                host='localhost',
                user='adsb_user',
                password='secure_password',
                database='aircraft_monitoring'
            )
            self.create_tables(conn)
            return conn
        except Exception as e:
            print(f"Database error: {e}")
            return None
    
    def create_tables(self, conn):
        """إنشاء جداول قاعدة البيانات"""
        cursor = conn.cursor()
        
        # جدول الطائرات النشطة
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aircraft (
                hex_id VARCHAR(10) PRIMARY KEY,
                flight VARCHAR(10),
                latitude FLOAT,
                longitude FLOAT,
                altitude INT,
                speed INT,
                heading INT,
                squawk VARCHAR(4),
                last_seen TIMESTAMP
            )
        """)
        
        # جدول سجل الرحلات
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flight_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                hex_id VARCHAR(10),
                flight VARCHAR(10),
                latitude FLOAT,
                longitude FLOAT,
                altitude INT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    
    def start_receiver(self):
        """بدء استقبال البيانات"""
        self.running = True
        thread = threading.Thread(target=self.receive_data)
        thread.daemon = True
        thread.start()
        print(f"[*] ADS-B Receiver started on {self.host}:{self.port}")
    
    def receive_data(self):
        """استقبال البيانات من dump1090"""
        while self.running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.connect((self.host, self.port))
                    sock.settimeout(10)
                    
                    while self.running:
                        data = sock.recv(1024).decode('utf-8')
                        if data:
                            self.process_message(data)
            except Exception as e:
                print(f"Connection error: {e}")
                time.sleep(5)
    
    def process_message(self, message):
        """معالجة رسائل ADS-B"""
        fields = message.split(',')
        
        if len(fields) < 22:
            return
        
        message_type = fields[1]
        
        if message_type == '3':  # رسالة تحديد الهوية والموقع
            hex_id = fields[4].strip()
            flight = fields[10].strip()
            altitude = int(fields[11]) if fields[11] else 0
            speed = int(fields[12]) if fields[12] else 0
            heading = int(fields[13]) if fields[13] else 0
            lat = float(fields[14]) if fields[14] else None
            lon = float(fields[15]) if fields[15] else None
            squawk = fields[17] if fields[17] else '0000'
            
            aircraft_info = {
                'hex_id': hex_id,
                'flight': flight,
                'latitude': lat,
                'longitude': lon,
                'altitude': altitude,
                'speed': speed,
                'heading': heading,
                'squawk': squawk,
                'last_update': datetime.now()
            }
            
            # تحديث البيانات في الذاكرة
            self.aircraft_data[hex_id] = aircraft_info
            
            # حفظ في قاعدة البيانات
            self.save_to_database(aircraft_info)
            
            # طباعة معلومات أساسية
            if flight and lat and lon:
                print(f"[{datetime.now()}] {flight} - Alt: {altitude}ft - Speed: {speed}kt")
    
    def save_to_database(self, data):
        """حفظ البيانات في MySQL"""
        if not self.db_connection:
            return
        
        try:
            cursor = self.db_connection.cursor()
            
            # تحديث أو إدراج بيانات الطائرة
            cursor.execute("""
                INSERT INTO aircraft 
                (hex_id, flight, latitude, longitude, altitude, speed, heading, squawk, last_seen)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                flight = VALUES(flight),
                latitude = VALUES(latitude),
                longitude = VALUES(longitude),
                altitude = VALUES(altitude),
                speed = VALUES(speed),
                heading = VALUES(heading),
                squawk = VALUES(squawk),
                last_seen = VALUES(last_seen)
            """, (
                data['hex_id'], data['flight'], data['latitude'], 
                data['longitude'], data['altitude'], data['speed'],
                data['heading'], data['squawk'], data['last_update']
            ))
            
            # حفظ في سجل التاريخ
            if data['latitude'] and data['longitude']:
                cursor.execute("""
                    INSERT INTO flight_history 
                    (hex_id, flight, latitude, longitude, altitude)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    data['hex_id'], data['flight'], data['latitude'],
                    data['longitude'], data['altitude']
                ))
            
            self.db_connection.commit()
        except Exception as e:
            print(f"Database save error: {e}")

    def get_active_aircraft(self):
        """الحصول على الطائرات النشطة"""
        # تنظيف الطائرات القديمة (أكثر من 5 دقائق)
        cutoff = datetime.now().timestamp() - 300
        self.aircraft_data = {
            k: v for k, v in self.aircraft_data.items()
            if v['last_update'].timestamp() > cutoff
        }
        return self.aircraft_data

# ------------------------------------------------------------
# واجهة ويب للعرض
# ------------------------------------------------------------

app = Flask(__name__)
receiver = ADSBReceiver()

@app.route('/')
def dashboard():
    """لوحة التحكم الرئيسية"""
    aircrafts = receiver.get_active_aircraft()
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ADS-B Aircraft Monitor</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/leaflet.css" />
        <style>
            body { margin: 0; padding: 20px; font-family: Arial; background: #0f172a; color: #fff; }
            .container { display: flex; height: 90vh; }
            #map { flex: 3; border-radius: 10px; }
            .sidebar { flex: 1; background: #1e293b; padding: 20px; margin-left: 20px; border-radius: 10px; }
            .aircraft-card { background: #334155; padding: 10px; margin: 10px 0; border-radius: 5px; }
            h2 { color: #60a5fa; }
        </style>
    </head>
    <body>
        <h1>✈️ نظام مراقبة الطائرات التجارية</h1>
        <div class="container">
            <div id="map"></div>
            <div class="sidebar">
                <h2>الطائرات النشطة: <span id="count">{{ count }}</span></h2>
                <div id="aircraft-list">
                    {% for aircraft in aircrafts.values() %}
                    <div class="aircraft-card">
                        <strong>{{ aircraft.flight or 'N/A' }}</strong><br>
                        ارتفاع: {{ aircraft.altitude }} قدم<br>
                        سرعة: {{ aircraft.speed }} عقدة<br>
                        Squawk: {{ aircraft.squawk }}
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/leaflet.js"></script>
        <script>
            var map = L.map('map').setView([24.7136, 46.6753], 6); // مركز السعودية
            
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap'
            }).addTo(map);
            
            // إضافة علامات للطائرات
            {% for aircraft in aircrafts.values() %}
                {% if aircraft.latitude and aircraft.longitude %}
                    L.marker([{{ aircraft.latitude }}, {{ aircraft.longitude }}])
                        .bindPopup("<b>{{ aircraft.flight }}</b><br>ارتفاع: {{ aircraft.altitude }} قدم")
                        .addTo(map);
                {% endif %}
            {% endfor %}
            
            // تحديث تلقائي كل 5 ثواني
            setInterval(function() {
                fetch('/data')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('count').textContent = Object.keys(data).length;
                        // يمكن إضافة تحديث الخريطة هنا
                    });
            }, 5000);
        </script>
    </body>
    </html>
    """
    
    return render_template_string(html_template, 
                                  aircrafts=aircrafts, 
                                  count=len(aircrafts))

@app.route('/data')
def get_data():
    """API للحصول على البيانات كـ JSON"""
    return json.dumps(receiver.get_active_aircraft(), 
                      default=str, 
                      ensure_ascii=False)

@app.route('/alerts')
def check_alerts():
    """فحص الإنذارات"""
    alerts = []
    for hex_id, data in receiver.aircraft_data.items():
        # تنبيه للطائرات المنخفضة
        if data.get('altitude', 0) < 1000 and data.get('altitude', 0) > 0:
            alerts.append(f"{data.get('flight', hex_id)} - منخفضة جداً: {data['altitude']} قدم")
        
        # تنبيه لرمز Squawk الطارئ
        emergency_squawks = ['7700', '7600', '7500']
        if data.get('squawk') in emergency_squawks:
            alerts.append(f"🚨 {data.get('flight', hex_id)} - رمز طارئ: {data['squawk']}")
    
    return json.dumps({'alerts': alerts})

# ------------------------------------------------------------
# التشغيل الرئيسي
# ------------------------------------------------------------

if __name__ == "__main__":
    # بدء مستقبل ADS-B
    receiver.start_receiver()
    
    # بدء خادم الويب
    print("[*] Starting web dashboard on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
