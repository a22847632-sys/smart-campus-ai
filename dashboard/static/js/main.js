document.addEventListener('DOMContentLoaded', () => {
    // Live Clock Update
    function updateClock() {
        const now = new Date();
        document.getElementById('live-clock').innerText = now.toLocaleTimeString();
    }
    setInterval(updateClock, 1000);
    updateClock();

    // Fetch Dashboard Telemetry Data
    async function fetchTelemetry() {
        try {
            const res = await fetch('/api/telemetry');
            if (!res.ok) return;
            const data = await res.json();
            
            updateOverviewKPIs(data);
            updateParkingWidget(data.parking);
            updateCrowdWidget(data.crowd);
            updateAlertsTable(data.alerts);
        } catch (err) {
            console.error("Error fetching telemetry:", err);
        }
    }

    function updateOverviewKPIs(data) {
        // Active Cams
        document.getElementById('kpi-cams').innerText = `${data.active_cameras || 2} / 2`;

        // Parking KPI
        const p = data.parking || {};
        const total = p.total_slots || 10;
        const avail = p.available_slots !== undefined ? p.available_slots : 10;
        const occ = p.occupied_slots || 0;
        const occRate = p.occupancy_rate || 0.0;

        document.getElementById('kpi-parking').innerText = `${avail} FREE`;
        document.getElementById('kpi-parking-sub').innerText = `${occ} / ${total} Occupied (${occRate.toFixed(1)}%)`;

        // Crowd KPI
        const c = data.crowd || {};
        const crowdStatus = c.crowd_status || 'LOW';
        const queueCount = c.queue_count || 0;
        
        document.getElementById('kpi-crowd').innerText = crowdStatus;
        document.getElementById('kpi-crowd-sub').innerText = `Queue: ${queueCount} in line (${c.queue_status || 'NORMAL'})`;

        // Emergency Alerts KPI
        const alerts = data.alerts || [];
        const alertCount = alerts.length;
        const alertVal = document.getElementById('kpi-alerts');
        const alertSub = document.getElementById('kpi-alerts-sub');

        if (alertCount > 0) {
            alertVal.innerText = `${alertCount} ACTIVE`;
            alertVal.className = "kpi-value text-danger";
            alertSub.innerText = `Latest: ${alerts[0].event_type}`;
        } else {
            alertVal.innerText = "0 ACTIVE";
            alertVal.className = "kpi-value text-success";
            alertSub.innerText = "No critical incidents";
        }
    }

    function updateParkingWidget(p) {
        if (!p) return;
        const total = p.total_slots || 10;
        const occ = p.occupied_slots || 0;
        const avail = p.available_slots !== undefined ? p.available_slots : 10;
        const occRate = p.occupancy_rate || 0.0;

        document.getElementById('parking-total').innerText = total;
        document.getElementById('parking-occupied').innerText = occ;
        document.getElementById('parking-available').innerText = avail;

        const bar = document.getElementById('parking-bar');
        bar.style.width = `${occRate.toFixed(1)}%`;

        const badge = document.getElementById('parking-rate-badge');
        badge.innerText = `${occRate.toFixed(1)}% Occupied`;
        if (occRate > 80) {
            badge.className = "badge badge-danger";
        } else if (occRate > 50) {
            badge.className = "badge badge-info";
        } else {
            badge.className = "badge badge-success";
        }
    }

    function updateCrowdWidget(c) {
        if (!c) return;
        document.getElementById('crowd-count').innerText = c.total_people || 0;
        document.getElementById('crowd-density').innerText = c.crowd_status || 'LOW';
        document.getElementById('queue-status').innerText = `${c.queue_count || 0} (${c.queue_status || 'LOW'})`;

        const badge = document.getElementById('crowd-status-badge');
        badge.innerText = c.crowd_status || 'LOW';
        if (c.crowd_status === 'HIGH') {
            badge.className = "badge badge-danger";
        } else if (c.crowd_status === 'MEDIUM') {
            badge.className = "badge badge-info";
        } else {
            badge.className = "badge badge-success";
        }
    }

    function updateAlertsTable(alerts) {
        const tbody = document.getElementById('alerts-table-body');
        if (!alerts || alerts.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No emergency incidents recorded. System monitoring normally.</td></tr>`;
            return;
        }

        tbody.innerHTML = alerts.map((a, idx) => `
            <tr>
                <td>#${a.id || idx + 1}</td>
                <td><span class="badge badge-danger">${a.event_type}</span></td>
                <td>${a.location}</td>
                <td>${a.timestamp}</td>
                <td>${((a.confidence || 0.9) * 100).toFixed(0)}%</td>
                <td><span class="badge badge-danger">${a.status}</span></td>
                <td>${a.details || 'N/A'}</td>
            </tr>
        `).join('');
    }

    document.getElementById('refresh-alerts-btn').addEventListener('click', fetchTelemetry);

    // Initial fetch and 2-second interval polling
    fetchTelemetry();
    setInterval(fetchTelemetry, 2000);
});
