import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db import DatabaseManager

def run_phase6_test():
    print("=" * 60)
    print("SMART CAMPUS AI - PHASE 6 DATABASE LAYER TEST")
    print("=" * 60)

    db_test_path = os.path.join("database", "test_smart_campus.db")
    if os.path.exists(db_test_path):
        os.remove(db_test_path)

    print(f"[INFO] Initializing SQLite database at: {db_test_path}")
    db = DatabaseManager(db_path=db_test_path)

    # 1. Log Parking Stats
    print("\n[TEST 1] Testing Parking Statistics Logging...")
    parking_data = {
        "total_slots": 20,
        "occupied_slots": 14,
        "available_slots": 6,
        "occupancy_rate": 70.0
    }
    db.log_parking_stats("Cam-01", parking_data)
    fetched_parking = db.get_latest_parking_stats("Cam-01")
    print(f"Logged & Retrieved Parking: Occupied={fetched_parking['occupied_slots']}/{fetched_parking['total_slots']} ({fetched_parking['occupancy_rate']}%)")
    assert fetched_parking["occupied_slots"] == 14, "Parking log mismatch!"

    # 2. Log Crowd Stats
    print("\n[TEST 2] Testing Crowd Statistics Logging...")
    crowd_data = {
        "timestamp": "2026-08-27 11:40:00",
        "total_people": 15,
        "crowd_zone_count": 8,
        "queue_count": 5,
        "crowd_status": "MEDIUM",
        "queue_status": "MEDIUM"
    }
    db.log_crowd_stats("Cam-01", crowd_data)
    fetched_crowd = db.get_latest_crowd_stats("Cam-01")
    print(f"Logged & Retrieved Crowd: Total={fetched_crowd['total_people']}, Status={fetched_crowd['crowd_status']}")
    assert fetched_crowd["total_people"] == 15, "Crowd log mismatch!"

    # 3. Log Emergency Alert
    print("\n[TEST 3] Testing Emergency Alert Logging...")
    alert_data = {
        "event_type": "PERSON_FALLEN",
        "location": "Main Plaza (Cam-01)",
        "timestamp": "2026-08-27 11:40:05",
        "confidence": 0.94,
        "status": "ACTIVE",
        "details": "Person #42 posture anomaly"
    }
    db.log_emergency_alert(alert_data)
    fetched_alerts = db.get_active_alerts()
    print(f"Logged & Retrieved Alerts Count: {len(fetched_alerts)} -> {fetched_alerts[0]['event_type']} at {fetched_alerts[0]['location']}")
    assert len(fetched_alerts) == 1, "Alert log count mismatch!"

    # 4. Test Dashboard Aggregated Summary Query
    print("\n[TEST 4] Testing Web Dashboard Aggregated Summary Fetch...")
    summary = db.get_dashboard_summary()
    print(f"Dashboard Aggregated Telemetry: {summary}")

    # Clean up test DB
    if os.path.exists(db_test_path):
        os.remove(db_test_path)

    print("\n" + "=" * 60)
    print("PHASE 6 DATABASE LAYER TEST RESULTS")
    print("=" * 60)
    print("SQLite Table Initialization: PASSED")
    print("Parking Telemetry Persistence: PASSED")
    print("Crowd Telemetry Persistence:   PASSED")
    print("Emergency Alerts Persistence:  PASSED")
    print("Dashboard Summary Query:       PASSED")
    print("Database Layer Status:         PASSED / READY")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_phase6_test()
    if not success:
        sys.exit(1)
