"""User objective set: persistence, validation, the resolver for old results, calibrations across edits, presets."""

import json

import numpy as np
import pytest

from microtopos import export
from microtopos import objectives as objlib
from microtopos.calibration import CalibrationStore
from microtopos.objectives import LEITZ_NPL_PRESET, ObjectiveError, ObjectiveSet
from microtopos.optics import Objective


@pytest.fixture(autouse=True)
def _restore_active():
    yield
    objlib.set_active(objlib.default_set())


def test_default_without_file(tmp_path):
    s = objlib.load_set(tmp_path)
    assert s.names == ["4x / 0.10", "10x / 0.25", "40x / 0.85", "100x / 1.25 oil"]
    assert not s.load_error


def test_round_trip(tmp_path):
    s = objlib.preset_set(LEITZ_NPL_PRESET)
    s.objectives.append(Objective("Custom 7x", 7.0, 0.15, 1.0, "0.17", "Acme", "Plan"))
    s.revision = 3
    objlib.save_set(tmp_path, s)
    back = objlib.load_set(tmp_path)
    assert back == s and not back.load_error
    raw = json.loads((tmp_path / "objectives.json").read_text(encoding="utf-8"))
    assert raw["schema"] == 1 and raw["source"] == LEITZ_NPL_PRESET and raw["revision"] == 3 and raw["date"]
    assert raw["objectives"][-1] == {"name": "Custom 7x", "magnification": 7.0, "na": 0.15, "immersion_n": 1.0,
                                     "cover": "0.17", "maker": "Acme", "series": "Plan"}


def test_leitz_preset_values():
    objs = {o.name: o for o in objlib.PRESETS[LEITZ_NPL_PRESET]}
    want = {"NPL 5x / 0.09": (5, 0.09, "∞/-"), "NPL 10x / 0.20": (10, 0.20, "∞/-"),
            "NPL 20x / 0.40": (20, 0.40, "∞/-"), "NPL Fluotar 50x / 0.85": (50, 0.85, "∞/0")}
    assert set(objs) == set(want)
    for name, (m, na, cover) in want.items():
        o = objs[name]
        assert (o.magnification, o.na, o.cover, o.immersion_n) == (m, na, cover, 1.0)  # all dry
        assert o.maker == "Leitz Wetzlar" and o.series.startswith("NPL")
    assert [o.magnification for o in objlib.PRESETS[LEITZ_NPL_PRESET]] == [5, 10, 20, 50]  # nosepiece order
    assert not objlib.problems(list(objlib.PRESETS[LEITZ_NPL_PRESET]))


@pytest.mark.parametrize("objs, fragment", [
    ([Objective("a", 10, 1.2)], "exceeds the immersion index"),  # dry NA > 1
    ([Objective("a", 10, 0.0)], "NA must be greater than 0"),
    ([Objective("a", 10, -0.1)], "NA must be greater than 0"),
    ([Objective("a", 0, 0.2)], "magnification 0 is outside"),
    ([Objective("a", -5, 0.2)], "magnification -5 is outside"),
    ([Objective("a", 10, 0.2), Objective("A", 20, 0.4)], "already used"),
    ([Objective("", 10, 0.2)], "name is empty"),
    ([Objective("a|b", 10, 0.2)], "may not contain"),
    ([Objective("a", float("nan"), 0.2)], "not a finite number"),
    ([Objective("a", 10, 0.2, immersion_n=2.5)], "immersion index n 2.5"),
    ([], "at least one objective"),
])
def test_validation_failures(objs, fragment, tmp_path):
    assert any(fragment in p for p in objlib.problems(objs)), objlib.problems(objs)
    with pytest.raises(ObjectiveError):
        objlib.save_set(tmp_path, ObjectiveSet(objs))
    assert not (tmp_path / "objectives.json").exists()


def test_oil_na_above_one_is_valid():
    assert not objlib.problems([Objective("100x oil", 100, 1.25, 1.515)])


def test_damaged_file_falls_back_and_is_kept(tmp_path):
    p = tmp_path / "objectives.json"
    p.write_text(json.dumps({"schema": 1, "objectives": [{"name": "x", "magnification": 10, "na": 3.0}]}))
    s = objlib.load_set(tmp_path)
    assert s.names == objlib.default_set().names and "exceeds the immersion index" in s.load_error
    assert "3.0" in p.read_text()  # not overwritten


def _old_session(tmp_path, objective, spec=None):
    h = np.zeros((8, 8), np.float32)
    r = {"height": h, "height_raw": h, "edf": np.zeros((8, 8), np.uint8), "um_per_px": 1.0, "z_unit": "um",
         "objective": objective}
    if spec:
        r["objective_spec"] = spec
    export.save_session(tmp_path / "s.npz", r, {"objective": objective})
    return export.load_session(tmp_path / "s.npz")[0]


def test_resolver_old_session_with_removed_objective(tmp_path):
    objlib.set_active(objlib.preset_set(LEITZ_NPL_PRESET))  # the built-in 40x is no longer in the list
    r = _old_session(tmp_path, "40x / 0.85")  # saved by v0.8: no objective_spec
    o = objlib.resolve_result(r)
    assert (o.magnification, o.na) == (40.0, 0.85)
    # a removed custom objective: the result's own record of its optics is used
    spec = Objective("Custom 7x", 7.0, 0.15).to_dict()
    assert objlib.resolve_result(_old_session(tmp_path, "Custom 7x", spec)).na == 0.15
    # neither in the list nor in a preset nor recorded: a clear error, not a KeyError
    with pytest.raises(ObjectiveError, match="not in the objective list.*Edit objectives"):
        objlib.resolve_result(_old_session(tmp_path, "Mystery 63x"))
    with pytest.raises(ObjectiveError, match="does not say which objective"):
        objlib.resolve(None)


def test_resolver_prefers_recorded_spec_over_edited_list():
    objlib.set_active(ObjectiveSet([Objective("10x / 0.25", 10, 0.30)]))  # user edited the NA since
    r = {"objective": "10x / 0.25", "objective_spec": Objective("10x / 0.25", 10, 0.25).to_dict()}
    assert objlib.resolve_result(r).na == 0.25
    assert objlib.resolve("10x / 0.25").na == 0.30


def _calibrate(store, name, lateral=0.5):
    c = store.get(name)
    c.lateral_um_per_px_full, c.lateral_source = lateral, "stage micrometer"
    c.darks["640x360@300"] = np.ones((4, 4), np.float32)
    store.touch(name)
    store.save()


def test_calibration_kept_across_edits(tmp_path):
    store = CalibrationStore(tmp_path)
    old = objlib.load_set(tmp_path)
    _calibrate(store, "40x / 0.85", 0.075)
    # switch to the Leitz preset: the 40x leaves the list, its calibration stays on disk
    new, notes = objlib.commit_edit(tmp_path, store, old, list(objlib.PRESETS[LEITZ_NPL_PRESET]),
                                    source=LEITZ_NPL_PRESET)
    assert objlib.active() == new and new.revision == old.revision + 1
    assert any("40x / 0.85" in n and "kept" in n for n in notes)
    again = CalibrationStore(tmp_path)
    assert again.get("40x / 0.85").lateral_um_per_px_full == 0.075
    assert "640x360@300" in again.get("40x / 0.85").darks
    # add it back: the calibration is there
    objs = new.objectives + [Objective("40x / 0.85", 40, 0.85)]
    objlib.commit_edit(tmp_path, again, new, objs)
    assert CalibrationStore(tmp_path).get("40x / 0.85").lateral_um_per_px_full == 0.075
    assert objlib.load_set(tmp_path).names[-1] == "40x / 0.85"


def test_rename_carries_calibration_and_conflict_is_refused(tmp_path):
    store = CalibrationStore(tmp_path)
    old = objlib.preset_set(LEITZ_NPL_PRESET)
    objlib.save_set(tmp_path, old)
    _calibrate(store, "NPL 10x / 0.20", 0.3)
    objs = [Objective("Leitz 10x", 10, 0.20) if o.name == "NPL 10x / 0.20" else o for o in old.objectives]
    new, _ = objlib.commit_edit(tmp_path, store, old, objs, {"NPL 10x / 0.20": "Leitz 10x"})
    back = CalibrationStore(tmp_path)
    assert back.get("Leitz 10x").lateral_um_per_px_full == 0.3 and not back.has_data("NPL 10x / 0.20")
    # renaming onto a name that holds another calibration (a removed objective's) is refused, nothing changes
    _calibrate(back, "Old 20x", 0.15)
    objs2 = [Objective("Old 20x", 20, 0.40) if o.name == "NPL 20x / 0.40" else o for o in new.objectives]
    _calibrate(back, "NPL 20x / 0.40", 0.16)
    with pytest.raises(ObjectiveError, match="already stored"):
        objlib.commit_edit(tmp_path, back, new, objs2, {"NPL 20x / 0.40": "Old 20x"})
    assert objlib.load_set(tmp_path) == new
    assert CalibrationStore(tmp_path).get("NPL 20x / 0.40").lateral_um_per_px_full == 0.16


def test_magnification_change_clears_lateral_only(tmp_path):
    store = CalibrationStore(tmp_path)
    old = objlib.default_set()
    _calibrate(store, "10x / 0.25", 0.3)
    store.get("10x / 0.25").z_um_per_frame = 0.5
    objs = [Objective(o.name, 20.0, o.na) if o.name == "10x / 0.25" else o for o in old.objectives]
    _, notes = objlib.commit_edit(tmp_path, store, old, objs)
    c = CalibrationStore(tmp_path).get("10x / 0.25")
    assert c.lateral_um_per_px_full is None and c.z_um_per_frame == 0.5 and c.darks
    assert any("cleared" in n for n in notes)


def test_batch_rejects_unknown_objective(tmp_path):
    from microtopos import batch
    f = tmp_path / "list.json"
    objlib.save_set(tmp_path, objlib.preset_set(LEITZ_NPL_PRESET))
    f.write_text((tmp_path / "objectives.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SystemExit, match="not in the objective list.*NPL 5x"):
        batch.main([str(tmp_path), str(tmp_path / "out"), "--objectives", str(f), "--objective", "40x / 0.85"])
    with pytest.raises(ObjectiveError):
        batch.run_batch(tmp_path, tmp_path / "out", "Nope", objset=objlib.load_set(tmp_path))
