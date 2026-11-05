"""The user's objective set: which objectives sit on the nosepiece, persisted next to calibration.json.

objectives.json (schema 1) holds the objectives in nosepiece order plus where the set came from (a preset or
the user's edits), the date of the last change and a revision that grows on every change. A missing file means
the built-in default set. A damaged or invalid file is never overwritten silently: the default set is used for
this run and `ObjectiveSet.load_error` says why.

Every lookup of an objective by name goes through `resolve()`, which also understands results and sessions made
with objectives that are no longer in the set: a result records its objective's optics (`objective_spec`), and
results from before that field existed fall back to the presets (the old built-in names). An unknown name raises
`ObjectiveError` with a message a user can act on, never a KeyError.

Calibrations stay keyed by objective name (calibration.py). Removing an objective keeps its calibration;
renaming carries the calibration with it (`commit_edit`).
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .optics import BUILTIN_OBJECTIVES, Objective

SCHEMA = 1
FILENAME = "objectives.json"
MAX_OBJECTIVES = 20
KEYED = 9  # keys 1..9 select the first nine objectives
MAG_RANGE = (0.5, 250.0)
N_RANGE = (1.0, 1.8)  # air .. high-index immersion
COVER_CHOICES = ("", "0.17", "∞/0", "∞/-", "0", "-")  # suggestions; any short text is accepted


class ObjectiveError(ValueError):
    """An objective that cannot be used: unknown name, invalid optics or a conflicting edit."""


def _o(name, mag, na, n=1.0, cover="", maker="", series=""):
    return Objective(name, float(mag), float(na), float(n), cover, maker, series)


DEFAULT_PRESET = "Generic (4x, 10x, 40x, 100x oil)"
LEITZ_NPL_PRESET = "Leitz NPL (5x, 10x, 20x, 50x Fluotar)"
PRESETS: dict[str, tuple[Objective, ...]] = {
    DEFAULT_PRESET: BUILTIN_OBJECTIVES,
    # Leitz Wetzlar NPL infinity objectives for reflected light, as engraved (dry; "∞/-": with or without cover
    # glass, "∞/0": uncovered specimens only). The tube factor is 1x, but the camera's own relay optics and pixel
    # pitch are unknown, so the lateral scale still needs a stage micrometer per objective.
    LEITZ_NPL_PRESET: (
        _o("NPL 5x / 0.09", 5, 0.09, cover="∞/-", maker="Leitz Wetzlar", series="NPL"),
        _o("NPL 10x / 0.20", 10, 0.20, cover="∞/-", maker="Leitz Wetzlar", series="NPL"),
        _o("NPL 20x / 0.40", 20, 0.40, cover="∞/-", maker="Leitz Wetzlar", series="NPL"),
        _o("NPL Fluotar 50x / 0.85", 50, 0.85, cover="∞/0", maker="Leitz Wetzlar", series="NPL FLUOTAR"),
    ),
}


def problems(objs: list[Objective]) -> list[str]:
    """Everything wrong with a candidate set, as sentences; empty when the set is usable."""
    out = []
    if not objs:
        out.append("The set needs at least one objective.")
    if len(objs) > MAX_OBJECTIVES:
        out.append(f"At most {MAX_OBJECTIVES} objectives are supported ({len(objs)} given).")
    seen: dict[str, str] = {}
    for k, o in enumerate(objs, 1):
        tag = f"Objective {k} ({o.name!r})" if o.name else f"Objective {k}"
        name = o.name.strip()
        if not name:
            out.append(f"{tag}: the name is empty.")
        elif name != o.name:
            out.append(f"{tag}: the name starts or ends with a space.")
        elif len(name) > 60:
            out.append(f"{tag}: the name is longer than 60 characters.")
        elif "|" in name:
            out.append(f"{tag}: the name may not contain '|'.")
        elif name.casefold() in seen:
            out.append(f"{tag}: the name is already used by {seen[name.casefold()]!r}.")
        else:
            seen[name.casefold()] = name
        for label, v in (("magnification", o.magnification), ("NA", o.na), ("immersion n", o.immersion_n)):
            if not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
                out.append(f"{tag}: {label} is not a finite number.")
                break
        else:
            if not (MAG_RANGE[0] <= o.magnification <= MAG_RANGE[1]):
                out.append(f"{tag}: magnification {o.magnification:g} is outside {MAG_RANGE[0]:g}–{MAG_RANGE[1]:g}.")
            if not (N_RANGE[0] <= o.immersion_n <= N_RANGE[1]):
                out.append(f"{tag}: immersion index n {o.immersion_n:g} is outside {N_RANGE[0]:g}–{N_RANGE[1]:g} "
                           "(1.0 for a dry objective, about 1.515 for oil).")
            if o.na <= 0:
                out.append(f"{tag}: NA must be greater than 0.")
            elif o.na > o.immersion_n:
                out.append(f"{tag}: NA {o.na:g} exceeds the immersion index n {o.immersion_n:g}; a dry objective "
                           "has NA below 1, an NA above 1 needs oil or water immersion.")
        if len(o.cover) > 12:
            out.append(f"{tag}: the cover-glass note is longer than 12 characters.")
    return out


def validate(objs: list[Objective]) -> None:
    p = problems(objs)
    if p:
        raise ObjectiveError(" ".join(p))


def _today() -> str:
    return datetime.date.today().isoformat()


@dataclass
class ObjectiveSet:
    objectives: list[Objective]
    source: str = DEFAULT_PRESET  # the preset it started from, "+ edited" once the user changed it
    date: str = ""
    revision: int = 0
    load_error: str = field(default="", compare=False)  # why objectives.json was not used (this run only)

    @property
    def names(self) -> list[str]:
        return [o.name for o in self.objectives]

    @property
    def magnifications(self) -> list[float]:
        return [o.magnification for o in self.objectives]

    def get(self, name: str) -> Objective | None:
        return next((o for o in self.objectives if o.name == name), None)

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "source": self.source, "date": self.date, "revision": self.revision,
                "objectives": [o.to_dict() for o in self.objectives]}

    @classmethod
    def from_json(cls, d: dict) -> "ObjectiveSet":
        if not isinstance(d, dict) or not isinstance(d.get("objectives"), list):
            raise ObjectiveError("not an objective list")
        if int(d.get("schema", 1)) > SCHEMA:
            raise ObjectiveError(f"objective list schema {d.get('schema')} is newer than this program ({SCHEMA})")
        try:
            objs = [Objective.from_dict(x) for x in d["objectives"]]
        except ValueError as e:
            raise ObjectiveError(str(e)) from None
        validate(objs)
        return cls(objs, str(d.get("source", "")), str(d.get("date", "")), int(d.get("revision", 0)))


def default_set() -> ObjectiveSet:
    return ObjectiveSet(list(BUILTIN_OBJECTIVES), DEFAULT_PRESET)


def preset_set(name: str) -> ObjectiveSet:
    if name not in PRESETS:
        raise ObjectiveError(f"unknown preset {name!r}; presets: {', '.join(PRESETS)}")
    return ObjectiveSet(list(PRESETS[name]), name, _today())


def load_set(directory: Path | None) -> ObjectiveSet:
    """The set stored in `directory`, the default set when there is none, or the default set with load_error set
    when the stored file cannot be used (the file itself is left alone)."""
    if directory is None:
        return default_set()
    path = Path(directory) / FILENAME
    if not path.exists():
        return default_set()
    try:
        return ObjectiveSet.from_json(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as e:  # ObjectiveError and JSONDecodeError are ValueErrors
        s = default_set()
        s.load_error = f"{path} was not used ({e}); the built-in objectives are active until you save a new list."
        return s


def save_set(directory: Path | None, objset: ObjectiveSet) -> None:
    """Validate and write atomically (temporary file, then replace)."""
    validate(objset.objectives)
    if directory is None:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".objectives-{os.getpid()}.tmp"
    tmp.write_text(json.dumps(objset.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(directory / FILENAME)


def default_data_dir() -> Path:
    """The folder the GUI keeps calibration.json and objectives.json in (Qt's AppDataLocation for MicroTopos)."""
    from PySide6.QtCore import QCoreApplication, QStandardPaths

    if not QCoreApplication.applicationName() or QCoreApplication.instance() is None:
        QCoreApplication.setOrganizationName("MicroTopos")
        QCoreApplication.setApplicationName("MicroTopos")
    loc = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    return Path(loc) if loc else Path.home() / ".microtopos"


# ---------------------------------------------------------------- the active set and the resolver
_active: ObjectiveSet = default_set()


def active() -> ObjectiveSet:
    return _active


def set_active(objset: ObjectiveSet) -> None:
    global _active
    validate(objset.objectives)
    _active = objset


def resolve(name: str | None, spec: dict | None = None, objset: ObjectiveSet | None = None) -> Objective:
    """The one way to turn an objective name into optics.

    Order: the optics a result recorded (`spec`, the objective it was actually made with), then the given or
    active set, then the presets (results from versions without a recorded spec used the built-in names).
    """
    if spec:
        try:
            o = Objective.from_dict(spec)
            if not problems([o]):
                return o
        except ValueError:
            pass  # a damaged record falls through to the name
    if not name:
        raise ObjectiveError("The result or session does not say which objective it was made with.")
    s = objset or _active
    o = s.get(name)
    if o is not None:
        return o
    for objs in PRESETS.values():
        o = next((x for x in objs if x.name == name), None)
        if o is not None:
            return o
    raise ObjectiveError(f"Objective {name!r} is not in the objective list ({', '.join(s.names)}) and the result "
                         "does not record its optics. Add an objective with exactly this name in step 2 "
                         "(Edit objectives…) to use this result.")


def resolve_result(r: dict) -> Objective:
    """Optics of the objective a result or session was made with."""
    return resolve(r.get("objective"), r.get("objective_spec"))


# ---------------------------------------------------------------- editing with calibrations
def commit_edit(directory: Path | None, store, old: ObjectiveSet, new_objs: list[Objective],
                renames: dict[str, str] | None = None, source: str | None = None) -> tuple[ObjectiveSet, list[str]]:
    """Validate a new objective list, carry calibrations across renames, save both files and make it active.

    `renames` maps an old objective name to its new name. A rename is refused (ObjectiveError) when a stored
    calibration already uses the new name and is not itself being renamed away. Calibrations of objectives that
    are dropped are kept. If an objective's magnification changes, its stored lateral scale is cleared, because
    the stage-micrometer value belonged to the old nameplate; dark/flat frames and the Z estimate are kept.
    Returns the new set and notes for the user.
    """
    validate(new_objs)
    renames = {a: b for a, b in (renames or {}).items() if a != b}
    names = [o.name for o in new_objs]
    for a, b in renames.items():
        if b not in names:
            raise ObjectiveError(f"rename of {a!r} to {b!r}: {b!r} is not in the new list")
    if len(set(renames.values())) != len(renames):
        raise ObjectiveError("two objectives are renamed to the same name")
    for a, b in renames.items():
        if store is not None and store.has_data(b) and b not in renames:
            raise ObjectiveError(f"Cannot rename {a!r} to {b!r}: a calibration is already stored under {b!r}. "
                                 "Choose another name, or remove that calibration first.")
    notes = []
    if store is not None:
        store.rename_many(renames)
        notes += [f"Calibration of {a!r} now belongs to {b!r}." for a, b in renames.items() if store.has_data(b)]
        old_by_name = {o.name: o for o in old.objectives}
        inverse = {b: a for a, b in renames.items()}
        for o in new_objs:
            prev = old_by_name.get(inverse.get(o.name, o.name))
            c = store.cals.get(o.name)
            if prev is not None and prev.magnification != o.magnification and c is not None \
                    and c.lateral_um_per_px_full:
                c.lateral_um_per_px_full, c.lateral_source, c.lateral_date = None, "", ""
                store.touch(o.name)
                notes.append(f"{o.name}: magnification changed from {prev.magnification:g}x to "
                             f"{o.magnification:g}x, so its lateral calibration was cleared; measure it again.")
        kept = [n for n in old.names if n not in names and n not in renames and store.has_data(n)]
        notes += [f"{n!r} was removed from the list; its calibration is kept and returns if you add it again."
                  for n in kept]
    src = source or (old.source if old.source.endswith("+ edited") else f"{old.source} + edited")
    new = ObjectiveSet(list(new_objs), src, _today(), old.revision + 1)
    save_set(directory, new)
    if store is not None:
        store.save()
    set_active(new)
    return new, notes
