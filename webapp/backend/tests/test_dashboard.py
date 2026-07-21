"""Dashboard aggregation tests."""


def test_dashboard_aggregates(auth_client, img_bytes):
    auth_client.post("/api/inspections", data={"mode": "adaptive", "imgsz": "640"},
                     files=[("files", ("a.jpg", img_bytes, "image/jpeg"))])
    r = auth_client.get("/api/dashboard/stats")
    assert r.status_code == 200
    s = r.json()
    assert s["total_inspections"] >= 1
    assert s["total_defects"] >= 1
    assert "crazing" in s["class_counts"]
    assert len(s["over_time"]) == 14
    assert s["mode_split"]["adaptive"] >= 1


def test_dashboard_requires_auth(client):
    assert client.get("/api/dashboard/stats").status_code == 401
