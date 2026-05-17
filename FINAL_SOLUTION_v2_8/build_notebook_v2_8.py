"""
Build script for V2.8 notebook.

Changes from V2.6:
  1. get_seg_mask() — Salamander yellow-spot + specular filter
       Flash reflections account for ~27% of SIFT/KAZE keypoints but carry
       no identity information.  Restricting keypoints to yellow-spot regions
       removes this noise.  Fallback to SAM3-only if no yellow pixels (<50 px).
  2. Preload cache — skip SalamanderID2025 files
       Features must be recomputed with the new mask.
  3. Calibration — add CALIB_KPT_CAP = 300
       Caps keypoints per image during calibration only (main pipeline unchanged).
       Reduces Salamander calibration from ~11,992s to ~500s.
"""

import json, copy, pathlib, sys

SRC = pathlib.Path("FINAL_SOLUTION_v2_6/ensemble_global_local_reid_v2_6.ipynb")
DST = pathlib.Path("FINAL_SOLUTION_v2_8/ensemble_global_local_reid_v2_8.ipynb")

assert SRC.exists(), f"Source notebook not found: {SRC}"

with open(SRC) as f:
    nb = json.load(f)

nb = copy.deepcopy(nb)

# ─── helper: patch cell source strings directly ──────────────────────────────
def patch_cell(nb, old, new, description):
    """Replace old→new in exactly one cell's source.  Works on plain Python strings."""
    count = 0
    for cell in nb["cells"]:
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        if old in src:
            count += 1
            src = src.replace(old, new, 1)
        cell["source"] = src   # store as single string (valid nbformat)
    assert count == 1, (
        f"Patch '{description}': expected 1 occurrence, found {count}\n"
        f"  looking for: {repr(old[:80])}"
    )

# ─── Patch 1: version comment ─────────────────────────────────────────────────
patch_cell(
    nb,
    "V2.3c: SeaTurtleID2022 and TexasHornedLizards use SIFT + KAZE (both via OpenCV)",
    "V2.8: yellow-spot + specular mask for SalamanderID2025; CALIB_KPT_CAP=300",
    "version comment",
)

# ─── Patch 2: preload cache — skip SalamanderID2025 ──────────────────────────
_PRELOAD_OLD = """\
                _src_f = os.path.join(_src_dir, _fname)
                _dst_f = os.path.join(_dst_dir, _fname)
                if not os.path.exists(_dst_f):
                    shutil.copy2(_src_f, _dst_f)
                    print(f"  Copied  {_sub}/{_fname}")
                    _copied += 1
                else:
                    print(f"  Present {_sub}/{_fname}")"""

_PRELOAD_NEW = """\
                # V2.8: Salamander features recomputed with yellow+specular mask
                if _fname.startswith('SalamanderID2025'):
                    print(f"  Skip    {_sub}/{_fname}  (V2.8 mask \u2192 recompute)")
                    continue
                _src_f = os.path.join(_src_dir, _fname)
                _dst_f = os.path.join(_dst_dir, _fname)
                if not os.path.exists(_dst_f):
                    shutil.copy2(_src_f, _dst_f)
                    print(f"  Copied  {_sub}/{_fname}")
                    _copied += 1
                else:
                    print(f"  Present {_sub}/{_fname}")"""

patch_cell(nb, _PRELOAD_OLD, _PRELOAD_NEW, "preload cache skip Salamander")

# ─── Patch 3: get_seg_mask — add yellow+specular filter for Salamander ────────
_SEG_OLD = """\
    # Background was painted to (255, 255, 255) — mark those as 0
    is_bg = (seg[:, :, 0] == 255) & (seg[:, :, 1] == 255) & (seg[:, :, 2] == 255)
    return (~is_bg).astype(np.uint8)

print('get_seg_mask() ready.')"""

_SEG_NEW = """\
    # Background was painted to (255, 255, 255) — mark those as 0
    is_bg = (seg[:, :, 0] == 255) & (seg[:, :, 1] == 255) & (seg[:, :, 2] == 255)
    mask  = (~is_bg).astype(np.uint8)
    # ── V2.8: Salamander yellow-spot + specular filter ────────────────────
    # Flash photography creates specular hotspots (~27% of SIFT kpts) that
    # carry no identity information.  Yellow spots are the actual per-
    # individual marker on fire salamanders.
    if 'SalamanderID2025' in rel_key:
        orig = cv2.imread(str(p))
        if orig is not None:
            hsv      = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
            specular = (hsv[:, :, 2] > 220) & (hsv[:, :, 1] < 40)
            yellow   = ((hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 42) &
                        (hsv[:, :, 1] >  80) & (hsv[:, :, 2] >  80))
            _k       = np.ones((25, 25), np.uint8)
            yellow_d = cv2.dilate(yellow.astype(np.uint8), _k).astype(bool)
            no_spec  = mask.astype(bool) & ~specular
            mask     = (no_spec & yellow_d).astype(np.uint8) if yellow_d.sum() >= 50 \\
                       else no_spec.astype(np.uint8)
    # ─────────────────────────────────────────────────────────────────────
    return mask

print('get_seg_mask() ready.')"""

patch_cell(nb, _SEG_OLD, _SEG_NEW, "get_seg_mask Salamander filter")

# ─── Patch 4: calibration — add CALIB_KPT_CAP constant ───────────────────────
_KPT_CAP_OLD = "CALIB_BFM_TOPK  = 50    # top-K preselection per image for BFMatcher\n\nGLOBAL_W_GRID"
_KPT_CAP_NEW = (
    "CALIB_BFM_TOPK  = 50    # top-K preselection per image for BFMatcher\n"
    "CALIB_KPT_CAP   = 300   # max keypoints/image during calibration (speed)\n"
    "\nGLOBAL_W_GRID"
)
patch_cell(nb, _KPT_CAP_OLD, _KPT_CAP_NEW, "CALIB_KPT_CAP constant")

# ─── Patch 5: calibration — SIFT_create uses CALIB_KPT_CAP ───────────────────
patch_cell(
    nb,
    "cv2.SIFT_create(nfeatures=1000)",
    "cv2.SIFT_create(nfeatures=CALIB_KPT_CAP)",
    "SIFT_create nfeatures",
)

# ─── Patch 6: calibration — cap KAZE descriptors before appending ─────────────
_KAZE_CAP_OLD = """\
            all_descs.append(descs.astype(np.float32))
        except Exception:
            all_descs.append(None)
    return all_descs"""

_KAZE_CAP_NEW = """\
            if len(descs) > CALIB_KPT_CAP:
                descs = descs[:CALIB_KPT_CAP]
            all_descs.append(descs.astype(np.float32))
        except Exception:
            all_descs.append(None)
    return all_descs"""

patch_cell(nb, _KAZE_CAP_OLD, _KAZE_CAP_NEW, "KAZE descriptor cap in calibration")

# ─── Patch 7: calibration — fix get_seg_mask call to use absolute path ────────
# In _extract_calib_local, `key` is a relative path (e.g. "images/Salamander/...").
# get_seg_mask does p.relative_to(_root) which raises ValueError for relative
# paths → always returns None → SAM3 mask AND yellow+specular filter never
# applied during calibration.  Fix: pass `path` (the absolute path) instead.
_SEG_KEY_OLD = "seg_mask = get_seg_mask(key)   # None if not in SAM3 cache"
_SEG_KEY_NEW = "seg_mask = get_seg_mask(path)  # absolute path → SAM3 + V2.8 yellow filter"
patch_cell(nb, _SEG_KEY_OLD, _SEG_KEY_NEW, "calibration get_seg_mask absolute path")

# ─── Patch 8: calibration title ───────────────────────────────────────────────
patch_cell(
    nb,
    "V2.6 -- Joint Weight + Threshold Calibration (Training Identities)",
    "V2.8 -- Joint Weight + Threshold Calibration (Training Identities)",
    "calibration title",
)

# ─── Write output ─────────────────────────────────────────────────────────────
with open(DST, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Written: {DST}  ({len(nb['cells'])} cells)")

# ─── Verification ─────────────────────────────────────────────────────────────
print("\nVerification:")

with open(DST) as f:
    out_text = f.read()

# Also join all cell sources for plain-text checks
all_src = ""
nb_out = json.loads(out_text)
for cell in nb_out["cells"]:
    s = cell["source"]
    all_src += (s if isinstance(s, str) else "".join(s)) + "\n"

ABSENT = set()  # patterns that must NOT be present

checks = [
    # Patch 1
    ("version comment updated",
     "V2.8: yellow-spot + specular mask for SalamanderID2025"),
    ("OLD version comment gone", "V2.3c: SeaTurtleID2022 and TexasHornedLizards use SIFT + KAZE (both via OpenCV)"),
    # Patch 2
    ("preload: skip Salamander branch",
     "if _fname.startswith('SalamanderID2025'):"),
    ("preload: recompute comment",
     "V2.8 mask"),
    ("preload: shutil.copy2 still present",
     "shutil.copy2(_src_f, _dst_f)"),
    # Patch 3
    ("get_seg_mask: mask variable",
     "mask  = (~is_bg).astype(np.uint8)"),
    ("get_seg_mask: SalamanderID2025 branch",
     "if 'SalamanderID2025' in rel_key:"),
    ("get_seg_mask: specular threshold",
     "specular = (hsv[:, :, 2] > 220) & (hsv[:, :, 1] < 40)"),
    ("get_seg_mask: yellow H range",
     "(hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 42)"),
    ("get_seg_mask: dilate kernel",
     "cv2.dilate(yellow.astype(np.uint8), _k).astype(bool)"),
    ("get_seg_mask: no_spec line",
     "no_spec  = mask.astype(bool) & ~specular"),
    ("get_seg_mask: fallback condition",
     "yellow_d.sum() >= 50"),
    ("get_seg_mask: return mask",
     "return mask"),
    ("get_seg_mask: print ready",
     "print('get_seg_mask() ready.')"),
    ("OLD return (~is_bg) gone",
     "return (~is_bg).astype(np.uint8)"),
    # Patch 4
    ("CALIB_KPT_CAP constant",
     "CALIB_KPT_CAP   = 300"),
    # Patch 5
    ("SIFT_create uses cap",
     "SIFT_create(nfeatures=CALIB_KPT_CAP)"),
    ("OLD SIFT_create gone",
     "SIFT_create(nfeatures=1000)"),
    # Patch 6
    ("KAZE cap check",
     "if len(descs) > CALIB_KPT_CAP:"),
    ("KAZE cap slice",
     "descs = descs[:CALIB_KPT_CAP]"),
    # Patch 7
    ("calibration: get_seg_mask uses absolute path",
     "seg_mask = get_seg_mask(path)  # absolute path"),
    ("calibration: OLD get_seg_mask(key) gone",
     "get_seg_mask(key)"),
    # Patch 8
    ("calibration title V2.8",
     "V2.8 -- Joint Weight + Threshold Calibration"),
    # Structure unchanged
    ("KAZEExtractor class present",
     "class KAZEExtractor"),
    ("SIFTExtractor class present",
     "class SIFTExtractor"),
    ("get_extractor function present",
     "def get_extractor"),
    ("calibration species loop",
     "for sp in CALIB_SPECIES:"),
    ("Cell 6.2 clustering present",
     "Cell 6.2: Generate Clusters"),
]

# Patterns that must be ABSENT
ABSENT = {
    "OLD version comment gone",
    "OLD return (~is_bg) gone",
    "OLD SIFT_create gone",
    "calibration: OLD get_seg_mask(key) gone",
}

passed = 0
failed = 0
for name, pattern in checks:
    present  = pattern in all_src
    must_absent = name in ABSENT
    ok = (not present) if must_absent else present
    mark = "✓" if ok else "✗"
    if not ok:
        failed += 1
        print(f"  {mark} FAIL [{name}]  {'(should be absent)' if must_absent else '(missing)'}: {repr(pattern[:70])}")
    else:
        passed += 1
        print(f"  {mark}  {name}")

print(f"\n{passed}/{passed+failed} checks passed")
if failed:
    sys.exit(1)
else:
    print("\nAll checks passed — V2.8 notebook ready.")
