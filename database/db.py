import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path=None):
        if db_path is None:
            db_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(db_dir, "smart_campus.db")
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name dict-style
        return conn

    def init_db(self):
        """Creates database schema tables if they do not already exist."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Parking Statistics Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parking_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                total_slots INTEGER NOT NULL,
                occupied_slots INTEGER NOT NULL,
                available_slots INTEGER NOT NULL,
                occupancy_rate REAL NOT NULL
            )
        """)

        # Crowd Statistics Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crowd_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                total_people INTEGER NOT NULL,
                crowd_zone_count INTEGER NOT NULL,
                queue_count INTEGER NOT NULL,
                crowd_status TEXT NOT NULL,
                queue_status TEXT NOT NULL
            )
        """)

        # Emergency Alerts Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emergency_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                location TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                details TEXT
            )
        """)

        # Camera Registry Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS camera_feeds (
                camera_id TEXT PRIMARY KEY,
                location_name TEXT NOT NULL,
                status TEXT NOT NULL,
                last_seen TEXT NOT NULL
            )
        """)

        # Register default cameras if missing
        cursor.execute("""
            INSERT OR IGNORE INTO camera_feeds (camera_id, location_name, status, last_seen)
            VALUES 
            ('Cam-01', 'Main Entrance & Parking', 'ONLINE', ?),
            ('Cam-02', 'Library Plaza & Courtyard', 'ONLINE', ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        conn.commit()
        conn.close()

    def log_parking_stats(self, camera_id, stats):
        """Inserts a new parking log entry."""
        conn = self.get_connection()
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO parking_logs (camera_id, timestamp, total_slots, occupied_slots, available_slots, occupancy_rate)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            camera_id,
            now_str,
            stats.get("total_slots", 0),
            stats.get("occupied_slots", 0),
            stats.get("available_slots", 0),
            stats.get("occupancy_rate", 0.0)
        ))
        conn.commit()
        conn.close()

    def log_crowd_stats(self, camera_id, analytics):
        """Inserts a new crowd log entry."""
        conn = self.get_connection()
        cursor = conn.cursor()
        now_str = analytics.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        cursor.execute("""
            INSERT INTO crowd_logs (camera_id, timestamp, total_people, crowd_zone_count, queue_count, crowd_status, queue_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            camera_id,
            now_str,
            analytics.get("total_people", 0),
            analytics.get("crowd_zone_count", 0),
            analytics.get("queue_count", 0),
            analytics.get("crowd_status", "LOW"),
            analytics.get("queue_status", "LOW")
        ))
        conn.commit()
        conn.close()

    def log_emergency_alert(self, alert_dict):
        """Inserts an emergency alert entry."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO emergency_alerts (event_type, location, timestamp, confidence, status, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            alert_dict.get("event_type", "UNKNOWN"),
            alert_dict.get("location", "Campus"),
            alert_dict.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            alert_dict.get("confidence", 0.0),
            alert_dict.get("status", "ACTIVE"),
            alert_dict.get("details", "")
        ))
        conn.commit()
        conn.close()

    def get_latest_parking_stats(self, camera_id="Cam-01"):
        """Returns the most recent parking stats record."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM parking_logs WHERE camera_id = ? ORDER BY id DESC LIMIT 1", (camera_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_latest_crowd_stats(self, camera_id="Cam-01"):
        """Returns the most recent crowd stats record."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM crowd_logs WHERE camera_id = ? ORDER BY id DESC LIMIT 1", (camera_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_active_alerts(self, limit=10):
        """Returns recent active emergency alerts."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM emergency_alerts ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_dashboard_summary(self):
        """Returns aggregated telemetry summary for the Web Dashboard API."""
        latest_parking = self.get_latest_parking_stats()
        latest_crowd = self.get_latest_crowd_stats()
        alerts = self.get_active_alerts(limit=5)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as active_cams FROM camera_feeds WHERE status = 'ONLINE'")
        active_cams = cursor.fetchone()["active_cams"]
        conn.close()

        return {
            "active_cameras": active_cams,
            "parking": latest_parking or {
                "total_slots": 10,
                "occupied_slots": 0,
                "available_slots": 10,
                "occupancy_rate": 0.0
            },
            "crowd": latest_crowd or {
                "total_people": 0,
                "crowd_status": "LOW",
                "queue_count": 0,
                "queue_status": "LOW"
            },
            "alerts": alerts
        }
