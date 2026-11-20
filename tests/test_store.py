"""Checks the 24-hour saved-scan store."""
from datetime import datetime, timedelta, timezone

from valuation import store

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


def saved(name, created=NOW):
    return store.SavedScan(name, ["AAPL"], [], created=created)


def test_save_list_newest_first():
    s = store.ScanStore()
    s.save("k", saved("first"), now=NOW)
    s.save("k", saved("second"), now=NOW)
    assert [x.name for x in s.list("k", now=NOW)] == ["second", "first"]


def test_scans_expire_after_24_hours():
    s = store.ScanStore()
    s.save("k", saved("old", created=NOW - timedelta(hours=25)), now=NOW - timedelta(hours=25))
    s.save("k", saved("today"), now=NOW)
    assert [x.name for x in s.list("k", now=NOW)] == ["today"]


def test_delete_and_get():
    s = store.ScanStore()
    a = saved("a")
    s.save("k", a, now=NOW)
    assert s.get("k", a.scan_id) is a
    s.delete("k", a.scan_id)
    assert s.list("k", now=NOW) == []


def test_keys_are_separate():
    s = store.ScanStore()
    s.save("one", saved("x"), now=NOW)
    assert s.list("two", now=NOW) == []


def test_running_scan_updates_saved_copy():
    s = store.ScanStore()
    rows = []
    s.save("k", store.SavedScan("live", ["AAPL", "KO"], rows), now=NOW)
    rows.append("AAPL result")                       # the scan keeps appending to the same list
    assert s.list("k", now=NOW)[0].done == 1


def test_keeps_at_most_20_per_key():
    s = store.ScanStore()
    for i in range(25):
        s.save("k", saved(f"s{i}"), now=NOW)
    assert len(s.list("k", now=NOW)) == store.MAX_SCANS_PER_KEY
