"""Auth endpoint tests."""


def test_signup_login_me(client):
    r = client.post("/auth/signup", json={"email": "a@b.io", "password": "password123", "full_name": "A"})
    assert r.status_code == 201
    body = r.json()
    assert body["user"]["email"] == "a@b.io"
    token = body["access_token"]

    r = client.post("/auth/login", json={"email": "a@b.io", "password": "password123"})
    assert r.status_code == 200

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "a@b.io"


def test_duplicate_email_conflicts(client):
    client.post("/auth/signup", json={"email": "d@b.io", "password": "password123", "full_name": ""})
    r = client.post("/auth/signup", json={"email": "d@b.io", "password": "password123", "full_name": ""})
    assert r.status_code == 409


def test_bad_login_is_401_with_error_shape(client):
    client.post("/auth/signup", json={"email": "e@b.io", "password": "password123", "full_name": ""})
    r = client.post("/auth/login", json={"email": "e@b.io", "password": "wrongpassword"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == 401


def test_me_requires_auth(client):
    assert client.get("/auth/me").status_code == 401


def test_short_password_rejected(client):
    r = client.post("/auth/signup", json={"email": "f@b.io", "password": "short", "full_name": ""})
    assert r.status_code == 422


def test_refresh_issues_new_token(client):
    tok = client.post("/auth/signup",
                      json={"email": "g@b.io", "password": "password123", "full_name": ""}).json()
    r = client.post("/auth/refresh", json={"refresh_token": tok["refresh_token"]})
    assert r.status_code == 200
    assert r.json()["access_token"]
