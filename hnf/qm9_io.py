"""QM9 XYZ loader (no torch_geometric / rdkit).

Downloads Figshare ``dsgdb9nsd.xyz.tar.bz2`` and caches a fixed subset as ``.npz``.

Property order in the XYZ comment line (after tag/index) follows Ramakrishnan 2014:
  A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset

QM9_URL = "https://ndownloader.figshare.com/files/3195389"
QM9_ARCHIVE_NAME = "dsgdb9nsd.xyz.tar.bz2"

# Z lookup
_ELEM = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
}

PROP_NAMES = (
    "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap",
    "r2", "zpve", "U0", "U", "H", "G", "Cv",
)


def _parse_float(tok: str) -> float:
    return float(tok.replace("*^", "e"))


def parse_qm9_xyz_text(text: str) -> dict:
    lines = text.strip().splitlines()
    n = int(lines[0].strip())
    # line 1: gdb_XXXXX  props...
    parts = lines[1].replace("\t", " ").split()
    # first token often "gdb" or similar; props are floats — find first float-like run
    props: list[float] = []
    for p in parts:
        try:
            props.append(_parse_float(p))
        except ValueError:
            continue
    # After index, there should be 15 properties; if index included as int first, drop it
    if len(props) >= 16:
        props = props[1:16]
    elif len(props) > 15:
        props = props[:15]
    if len(props) != 15:
        raise ValueError(f"expected 15 props, got {len(props)} from {parts[:5]}...")

    zs, pos = [], []
    for i in range(n):
        toks = lines[2 + i].replace("\t", " ").split()
        zs.append(_ELEM[toks[0]])
        pos.append([_parse_float(toks[1]), _parse_float(toks[2]), _parse_float(toks[3])])
    return {
        "z": np.asarray(zs, dtype=np.int64),
        "pos": np.asarray(pos, dtype=np.float32),
        "y": np.asarray(props, dtype=np.float32),
    }


def download_qm9_archive(data_dir: Path, url: str = QM9_URL) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / QM9_ARCHIVE_NAME
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"[qm9] archive exists {path}", flush=True)
        return path
    print(f"[qm9] downloading {url} → {path}", flush=True)
    urlretrieve(url, path)
    print(f"[qm9] downloaded {path.stat().st_size / 1e6:.1f} MB", flush=True)
    return path


def build_subset_cache(
    archive: Path,
    cache_path: Path,
    *,
    n_molecules: int = 12_000,
    seed: int = 42,
) -> Path:
    if cache_path.exists():
        print(f"[qm9] cache exists {cache_path}", flush=True)
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # reservoir / first pass: collect all member names then sample
    print(f"[qm9] indexing archive…", flush=True)
    with tarfile.open(archive, "r:bz2") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".xyz")]
        print(f"[qm9] {len(members)} xyz files", flush=True)
        if n_molecules >= len(members):
            chosen = members
        else:
            idx = rng.choice(len(members), size=n_molecules, replace=False)
            chosen = [members[i] for i in sorted(idx.tolist())]
        zs, pos, ys, natoms = [], [], [], []
        for i, m in enumerate(chosen):
            f = tar.extractfile(m)
            if f is None:
                continue
            text = f.read().decode("utf-8", errors="replace")
            try:
                mol = parse_qm9_xyz_text(text)
            except Exception as e:
                print(f"[qm9] skip {m.name}: {e}", flush=True)
                continue
            zs.append(mol["z"])
            pos.append(mol["pos"])
            ys.append(mol["y"])
            natoms.append(len(mol["z"]))
            if (i + 1) % 2000 == 0:
                print(f"[qm9] parsed {i+1}/{len(chosen)}", flush=True)
    max_n = int(max(natoms))
    n = len(ys)
    Z = np.zeros((n, max_n), dtype=np.int64)
    P = np.zeros((n, max_n, 3), dtype=np.float32)
    M = np.zeros((n, max_n), dtype=np.bool_)
    Y = np.stack(ys, axis=0)
    for i in range(n):
        k = natoms[i]
        Z[i, :k] = zs[i]
        P[i, :k] = pos[i]
        M[i, :k] = True
    np.savez_compressed(
        cache_path,
        z=Z, pos=P, mask=M, y=Y,
        prop_names=np.array(PROP_NAMES),
        n_molecules=n, max_n=max_n, seed=seed,
    )
    print(f"[qm9] wrote {cache_path} n={n} max_n={max_n}", flush=True)
    return cache_path


def ensure_qm9_subset(
    data_dir: Path,
    *,
    n_molecules: int = 12_000,
    seed: int = 42,
) -> Path:
    data_dir = Path(data_dir)
    cache = data_dir / f"qm9_subset_n{n_molecules}_s{seed}.npz"
    if cache.exists():
        return cache
    archive = download_qm9_archive(data_dir)
    return build_subset_cache(archive, cache, n_molecules=n_molecules, seed=seed)


class QM9SubsetDataset(Dataset):
    """Cached QM9 subset with fixed train/val/test split."""

    def __init__(
        self,
        cache_path: Path,
        split: str,
        *,
        target: str = "gap",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 42,
        standardize: bool = True,
        y_mean: Optional[float] = None,
        y_std: Optional[float] = None,
    ):
        z = np.load(cache_path, allow_pickle=False)
        self.Z = z["z"]
        self.pos = z["pos"]
        self.mask = z["mask"]
        self.Y_all = z["y"]
        names = [str(x) for x in z["prop_names"].tolist()]
        if target not in names:
            raise ValueError(f"target {target} not in {names}")
        self.target = target
        self.ti = names.index(target)
        y = self.Y_all[:, self.ti].astype(np.float64)

        n = len(y)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        if split == "train":
            idx = perm[:n_train]
        elif split == "val":
            idx = perm[n_train : n_train + n_val]
        elif split == "test":
            idx = perm[n_train + n_val :]
        else:
            raise ValueError(split)
        self.idx = idx

        train_y = y[perm[:n_train]]
        self.y_mean = float(train_y.mean()) if y_mean is None else float(y_mean)
        self.y_std = float(train_y.std() + 1e-8) if y_std is None else float(y_std)
        self.standardize = bool(standardize)
        self._y_raw = y

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict:
        j = int(self.idx[i])
        y = float(self._y_raw[j])
        if self.standardize:
            y = (y - self.y_mean) / self.y_std
        return {
            "pos": torch.from_numpy(self.pos[j]),
            "z": torch.from_numpy(self.Z[j]),
            "mask": torch.from_numpy(self.mask[j]),
            "y": torch.tensor(y, dtype=torch.float32),
            "y_raw": torch.tensor(float(self._y_raw[j]), dtype=torch.float32),
        }
