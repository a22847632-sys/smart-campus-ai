import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

def run_phase7_test():
    print("=" * 60)
    print("SMART CAMPUS AI - PHASE 7 WEB DASHBOARD TEST")
    print("=" * 60)

    client = app.test_client()

    # 1. Test Index Page Route
    print("\n[TEST 1] Testing Dashboard Root Route ('/')...")
    res = client.get('/')
    print(f"HTTP Status Code: {res.status_code}")
    assert res.status_code == 200, "Root route failed!"
    assert b"AI-POWERED SMART CAMPUS MANAGEMENT SYSTEM" in res.data, "HTML Title missing!"
    print("[SUCCESS] Dashboard HTML page rendered correctly.")

    # 2. Test Telemetry API Route
    print("\n[TEST 2] Testing Telemetry API ('/api/telemetry')...")
    res_tel = client.get('/api/telemetry')
    print(f"HTTP Status Code: {res_tel.status_code}")
    assert res_tel.status_code == 200, "Telemetry API failed!"
    json_data = res_tel.get_json()
    print(f"Telemetry Payload Keys: {list(json_data.keys())}")
    assert "active_cameras" in json_data, "active_cameras key missing!"
    assert "parking" in json_data, "parking key missing!"
    assert "crowd" in json_data, "crowd key missing!"
    print("[SUCCESS] Telemetry API response verified.")

    # 3. Test Parking API Route
    print("\n[TEST 3] Testing Parking API ('/api/parking')...")
    res_park = client.get('/api/parking')
    assert res_park.status_code == 200, "Parking API failed!"
    print(f"Parking API Payload: {res_park.get_json()}")
    print("[SUCCESS] Parking API verified.")

    # 4. Test Crowd API Route
    print("\n[TEST 4] Testing Crowd API ('/api/crowd')...")
    res_crowd = client.get('/api/crowd')
    assert res_crowd.status_code == 200, "Crowd API failed!"
    print(f"Crowd API Payload: {res_crowd.get_json()}")
    print("[SUCCESS] Crowd API verified.")

    # 5. Test Alerts API Route
    print("\n[TEST 5] Testing Alerts API ('/api/alerts')...")
    res_alerts = client.get('/api/alerts')
    assert res_alerts.status_code == 200, "Alerts API failed!"
    print(f"Alerts Count: {len(res_alerts.get_json())}")
    print("[SUCCESS] Alerts API verified.")

    print("\n" + "=" * 60)
    print("PHASE 7 WEB DASHBOARD TEST RESULTS")
    print("=" * 60)
    print("Flask Dashboard UI ('/'):          PASSED (HTTP 200)")
    print("Telemetry API ('/api/telemetry'):  PASSED (HTTP 200)")
    print("Parking API ('/api/parking'):      PASSED (HTTP 200)")
    print("Crowd API ('/api/crowd'):          PASSED (HTTP 200)")
    print("Alerts API ('/api/alerts'):        PASSED (HTTP 200)")
    print("Web Dashboard Status:              PASSED / READY")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_phase7_test()
    if not success:
        sys.exit(1)
