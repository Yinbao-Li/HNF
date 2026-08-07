#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download MPI-LEMON (Babayan 2019 Sci Data): resting EEG + T1/MP2RAGE.

Public GWDG mirror, no registration:
  https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/

Default pull (probe↔structure closure, not full MRI):
  - behavioural / META / availability
  - preprocessed EO+EC EEGLAB (.set/.fdt)
  - EEG digitised localizers
  - raw MP2RAGE UNI T1w + inv-2 (+ json / defacemask)
  - preprocessed skull-stripped brain (QC)

Full raw MRI (DWI/fMRI/T2/FLAIR) is ~2.5 GB/subject and is NOT downloaded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

BASE = "https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/"
HREF_RE = re.compile(r'href=[\'"]([^\'"]+)[\'"]', re.I)
UA = "HNF-LEMON-downloader/1.0 (+scientific reuse; Babayan 2019)"


class _DirParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-dir", default="external_data/eeg_lemon")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-subjects", type=int, default=0, help="0 = all overlapping EEG+T1")
    p.add_argument("--skip-eeg", action="store_true")
    p.add_argument("--skip-mri", action="store_true")
    p.add_argument("--skip-localizer", action="store_true")
    p.add_argument("--include-t2-flair", action="store_true")
    return p.parse_args()


def _open(url: str, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout)


def list_dir(url: str) -> list[str]:
    if not url.endswith("/"):
        url += "/"
    with _open(url, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    parser = _DirParser()
    parser.feed(html)
    out = []
    for href in parser.hrefs:
        if href in ("../", "./") or href.startswith("?") or href.startswith("#"):
            continue
        if href.startswith("http") and urlparse(href).netloc != urlparse(url).netloc:
            continue
        out.append(urljoin(url, href))
    # unique preserve order
    seen = set()
    uniq = []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def _content_length(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit():
                return int(cl)
    except Exception:
        return None
    return None


def download_file(url: str, dest: Path, retries: int = 4) -> tuple[str, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    expected_n = _content_length(url)

    if dest.is_file() and expected_n is not None and dest.stat().st_size == expected_n:
        return dest.name, "skip"
    if dest.is_file() and expected_n is None and dest.stat().st_size > 0:
        return dest.name, "skip-nolength"

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as fh:
                cl = resp.headers.get("Content-Length")
                expected_n = int(cl) if cl and cl.isdigit() else expected_n
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            if expected_n is not None and tmp.stat().st_size != expected_n:
                raise IOError(f"size mismatch {tmp.stat().st_size} != {expected_n}")
            tmp.replace(dest)
            return dest.name, "ok"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return dest.name, "missing"
            last_err = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            time.sleep(min(30, 2 ** attempt))
        except Exception as exc:  # noqa: BLE001 — resume loop
            last_err = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            time.sleep(min(30, 2 ** attempt))
    return dest.name, f"fail:{last_err}"


def rel_under(url: str, root: str) -> str:
    if not url.startswith(root):
        return urlparse(url).path.lstrip("/")
    return url[len(root) :]


def collect_behavioural(jobs: list[tuple[str, Path]], target: Path) -> None:
    beh_root = urljoin(BASE, "Behavioural_Data_MPILMBB_LEMON/")
    stack = [beh_root]
    while stack:
        url = stack.pop()
        for child in list_dir(url):
            if child.endswith("/"):
                stack.append(child)
            else:
                rel = rel_under(child, beh_root)
                jobs.append((child, target / "behavioural" / rel))
    mri_meta = urljoin(BASE, "MRI_MPILMBB_LEMON/")
    for name in (
        "Participants_LEMON.csv",
        "dataset_description.json",
        "participants_LSD_and_LEMON.tsv",
        "README",
    ):
        jobs.append((urljoin(mri_meta, name), target / "mri_meta" / name))
    jobs.append(
        (
            urljoin(BASE, "EEG_MPILMBB_LEMON/EEG_Info"),
            target / "eeg" / "EEG_Info",
        )
    )


def eeg_subject_ids(eeg_dir_url: str) -> list[str]:
    ids = set()
    for child in list_dir(eeg_dir_url):
        name = urlparse(child).path.split("/")[-1]
        m = re.match(r"(sub-\d+)_", name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def mri_subject_ids(mri_raw_url: str) -> list[str]:
    ids = []
    for child in list_dir(mri_raw_url):
        name = urlparse(child).path.rstrip("/").split("/")[-1]
        if name.startswith("sub-"):
            ids.append(name)
    return sorted(ids)


def collect_eeg(jobs: list[tuple[str, Path]], target: Path, subjects: set[str]) -> None:
    eeg_dir = urljoin(BASE, "EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/")
    for child in list_dir(eeg_dir):
        name = urlparse(child).path.split("/")[-1]
        m = re.match(r"(sub-\d+)_", name)
        if not m or m.group(1) not in subjects:
            continue
        if not (name.endswith(".set") or name.endswith(".fdt")):
            continue
        jobs.append((child, target / "eeg" / "preprocessed" / name))


def collect_localizer(jobs: list[tuple[str, Path]], target: Path, subjects: set[str]) -> None:
    loc_root = urljoin(BASE, "EEG_MPILMBB_LEMON/EEG_Localizer_BIDS_ID/")
    available = set()
    for child in list_dir(loc_root):
        if child.endswith("/"):
            available.add(urlparse(child).path.rstrip("/").split("/")[-1])
    for sid in sorted(subjects & available):
        for name in (f"{sid}.mat", "brainstormstudy.mat"):
            url = urljoin(loc_root, f"{sid}/{name}")
            jobs.append((url, target / "eeg" / "localizer" / sid / name))


def collect_mri(
    jobs: list[tuple[str, Path]],
    target: Path,
    subjects: set[str],
    include_t2_flair: bool,
) -> None:
    raw_root = urljoin(BASE, "MRI_MPILMBB_LEMON/MRI_Raw/")
    deriv_root = urljoin(BASE, "MRI_MPILMBB_LEMON/MRI_Preprocessed_Derivetives/")
    stems = [
        "acq-mp2rage_T1w.nii.gz",
        "acq-mp2rage_T1map.nii.gz",
        "acq-mp2rage_defacemask.nii.gz",
        "inv-2_mp2rage.nii.gz",
        "inv-2_mp2rage.json",
    ]
    if include_t2_flair:
        stems.extend(
            [
                "T2w.nii.gz",
                "T2w.json",
                "acq-lowres_FLAIR.nii.gz",
                "acq-lowres_FLAIR.json",
            ]
        )
    for sid in sorted(subjects):
        for stem in stems:
            name = f"{sid}_ses-01_{stem}"
            url = urljoin(raw_root, f"{sid}/ses-01/anat/{name}")
            jobs.append((url, target / "mri" / "raw" / sid / "ses-01" / "anat" / name))
        brain = f"{sid}_ses-01_acq-mp2rage_brain.nii.gz"
        jobs.append(
            (
                urljoin(deriv_root, f"{sid}/anat/{brain}"),
                target / "mri" / "derivatives" / sid / "anat" / brain,
            )
        )


def run_jobs(jobs: list[tuple[str, Path]], workers: int) -> dict[str, int]:
    counts = {"ok": 0, "skip": 0, "skip-nolength": 0, "missing": 0, "fail": 0}
    # dedupe dest
    uniq: dict[Path, str] = {}
    for url, dest in jobs:
        uniq[dest] = url
    items = list(uniq.items())
    print(f"[lemon] download {len(items)} files with {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(download_file, url, dest): dest for dest, url in items}
        done = 0
        for fut in as_completed(futs):
            name, status = fut.result()
            key = "fail" if status.startswith("fail") else status
            counts[key] = counts.get(key, 0) + 1
            done += 1
            if status.startswith("fail") or done % 25 == 0 or done == len(items):
                print(f"[lemon] {done}/{len(items)} {name} {status}", flush=True)
    return counts


def write_inventory(target: Path, eeg_ids: list[str], mri_ids: list[str], both: list[str]) -> None:
    inv = {
        "dataset": "MPI-LEMON",
        "citation": "Babayan et al. Sci Data 6, 180308 (2019). doi:10.1038/sdata.2018.308",
        "source": BASE,
        "license_note": "Public via GWDG / FCP-INDI / NITRC mplimbb. Cite descriptor paper.",
        "n_eeg_preprocessed": len(eeg_ids),
        "n_mri_raw": len(mri_ids),
        "n_eeg_and_mri": len(both),
        "eeg_ids": eeg_ids,
        "mri_ids": mri_ids,
        "eeg_and_mri_ids": both,
        "pulled": {
            "eeg": "preprocessed EO/EC EEGLAB .set/.fdt + localizer",
            "mri": "MP2RAGE UNI T1w + inv-2 + T1map + skullstripped brain",
            "not_pulled": "raw EEG (~54GB), DWI/fMRI/T2/FLAIR (~2.5GB/sub)",
        },
    }
    (target / "INVENTORY.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
    lines = [
        "participant_id\thas_eeg\thas_t1",
    ]
    all_ids = sorted(set(eeg_ids) | set(mri_ids))
    eeg_set, mri_set = set(eeg_ids), set(mri_ids)
    for sid in all_ids:
        lines.append(f"{sid}\t{int(sid in eeg_set)}\t{int(sid in mri_set)}")
    (target / "participants_overlap.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    target = Path(args.target_dir)
    if not target.is_absolute():
        target = _REPO / target
    target.mkdir(parents=True, exist_ok=True)

    print("[lemon] listing subject IDs…", flush=True)
    eeg_ids = eeg_subject_ids(
        urljoin(BASE, "EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/")
    )
    mri_ids = mri_subject_ids(urljoin(BASE, "MRI_MPILMBB_LEMON/MRI_Raw/"))
    both_all = sorted(set(eeg_ids) & set(mri_ids))
    both = both_all
    if args.max_subjects and args.max_subjects > 0:
        both = both_all[: int(args.max_subjects)]
    print(
        f"[lemon] eeg_prep={len(eeg_ids)} mri_raw={len(mri_ids)} "
        f"overlap={len(both_all)} pull={len(both)}",
        flush=True,
    )
    write_inventory(target, eeg_ids, mri_ids, both_all)

    jobs: list[tuple[str, Path]] = []
    print("[lemon] collect behavioural…", flush=True)
    collect_behavioural(jobs, target)
    subj = set(both)
    if not args.skip_eeg:
        print("[lemon] collect EEG…", flush=True)
        collect_eeg(jobs, target, subj)
        if not args.skip_localizer:
            print("[lemon] collect localizer…", flush=True)
            collect_localizer(jobs, target, subj)
    if not args.skip_mri:
        print("[lemon] collect MRI T1…", flush=True)
        collect_mri(jobs, target, subj, include_t2_flair=args.include_t2_flair)
    print(f"[lemon] queued {len(jobs)} urls", flush=True)

    counts = run_jobs(jobs, args.workers)
    print(f"[lemon] done counts={counts} → {target}", flush=True)
    (target / "DOWNLOAD_STATUS.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    if counts.get("fail", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
