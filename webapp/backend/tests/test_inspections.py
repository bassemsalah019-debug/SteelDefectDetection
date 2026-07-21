"""Inspection lifecycle tests (AI seam is mocked in conftest)."""


def _files(img_bytes):
    return [("files", ("sample.jpg", img_bytes, "image/jpeg"))]


def test_create_and_get(auth_client, img_bytes):
    r = auth_client.post("/api/inspections",
                         data={"title": "Scan A", "mode": "adaptive", "imgsz": "640"},
                         files=_files(img_bytes))
    assert r.status_code == 201, r.text
    d = r.json()
    assert d["n_images"] == 1
    assert d["n_defects"] == 1
    assert d["images"][0]["detections"][0]["cls_name"] == "crazing"
    assert d["images"][0]["annotated_url"].startswith("/media/")

    g = auth_client.get(f"/api/inspections/{d['id']}")
    assert g.status_code == 200
    assert g.json()["class_counts"] == {"crazing": 1}


def test_list_pagination(auth_client, img_bytes):
    auth_client.post("/api/inspections", data={"mode": "fixed", "imgsz": "640"}, files=_files(img_bytes))
    r = auth_client.get("/api/inspections?page=1&page_size=5")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1 and len(body["items"]) >= 1


def test_report_generation(auth_client, img_bytes):
    iid = auth_client.post("/api/inspections", data={"mode": "adaptive", "imgsz": "640"},
                           files=_files(img_bytes)).json()["id"]
    r = auth_client.post(f"/api/inspections/{iid}/report", json={"lang": "en"})
    assert r.status_code == 200
    assert "Summary" in r.json()["text"]


def test_delete(auth_client, img_bytes):
    iid = auth_client.post("/api/inspections", data={"mode": "adaptive", "imgsz": "640"},
                           files=_files(img_bytes)).json()["id"]
    assert auth_client.delete(f"/api/inspections/{iid}").status_code == 204
    assert auth_client.get(f"/api/inspections/{iid}").status_code == 404


def test_invalid_mode_rejected(auth_client, img_bytes):
    r = auth_client.post("/api/inspections", data={"mode": "bogus", "imgsz": "640"}, files=_files(img_bytes))
    assert r.status_code == 422


def test_requires_auth(client):
    assert client.get("/api/inspections").status_code == 401


def test_cannot_see_others_inspection(client, img_bytes):
    # user 1 creates an inspection
    t1 = client.post("/auth/signup", json={"email": "u1@b.io", "password": "password123", "full_name": ""}).json()
    client.headers.update({"Authorization": f"Bearer {t1['access_token']}"})
    iid = client.post("/api/inspections", data={"mode": "adaptive", "imgsz": "640"},
                      files=_files(img_bytes)).json()["id"]
    # user 2 must not be able to read it
    t2 = client.post("/auth/signup", json={"email": "u2@b.io", "password": "password123", "full_name": ""}).json()
    client.headers.update({"Authorization": f"Bearer {t2['access_token']}"})
    assert client.get(f"/api/inspections/{iid}").status_code == 404
