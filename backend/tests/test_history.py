from __future__ import annotations


def test_analysis_is_persisted_and_listed(client):
    assert client.get("/history").json()["total"] == 0
    posted = client.post("/analyze", json={"text": "Vaccines cause autism.",
                                           "url": "https://example.invalid/post"}).json()
    body = client.get("/history").json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["trust_score"] == posted["trust_score"]
    assert item["url"] == "https://example.invalid/post"
    assert item["text_excerpt"].startswith("Vaccines cause autism")


def test_history_is_newest_first_and_limited(client):
    for i in range(3):
        client.post("/analyze", json={"text": f"Claim number {i}."})
    items = client.get("/history?limit=2").json()["items"]
    assert len(items) == 2
    assert items[0]["text_excerpt"].endswith("2.")


def test_full_report_is_retrievable_by_id(client):
    client.post("/analyze", json={"text": "A claim."})
    analysis_id = client.get("/history").json()["items"][0]["id"]
    report = client.get(f"/report/{analysis_id}").json()
    assert report["explanation"]
    assert client.get("/report/99999").status_code == 404
