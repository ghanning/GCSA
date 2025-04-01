from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

SCENES_TRAIN = [
    "trondheim",
    "london",
    "boston",
    "melbourne",
    "amsterdam",
    "helsinki",
    "tokyo",
    "toronto",
    "saopaulo",
    "moscow",
    "zurich",
    "paris",
    "bangkok",
    "budapest",
    "austin",
    "berlin",
    "ottawa",
    "phoenix",
    "goa",
    "amman",
    "nairobi",
    "manila",
]
SCENES_VAL = ["cph", "sf"]
SCENES_TEST = ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"]
SCENES = SCENES_TRAIN + SCENES_VAL + SCENES_TEST

Metadata = namedtuple("Metadata", ["key", "lon", "lat", "ca", "captured_at", "pano"])
Subtasks = namedtuple("Subtasks", ["all", "w2s", "s2w", "d2n", "n2d", "o2n", "n2o"])


class MslsScene:
    """! Mapillary Street-level Sequences scene class.

    An instance of this class represents one scene (= city) of the dataset.
    """

    def __init__(
        self, name: str, dataset_dir: Path, exclude_panos: bool = False, subtask: Optional[str] = None
    ) -> None:
        """! Class initializer.

        @param name Scene name.
        @param dataset_dir Path to dataset root directory.
        @param exclude_panos Whether to exclude panorama images.
        @param subtask Subtask to consider.
        """
        self.name = name
        self.dataset_dir = dataset_dir

        self.query_mask, self.db_mask = None, None

        if exclude_panos or subtask:
            query_mask = np.ones(len(self.query_image_paths()), dtype=bool)
            db_mask = np.ones(len(self.db_image_paths()), dtype=bool)

            if exclude_panos:
                query_mask &= np.array([not m.pano for m in self.query_metadata()])
                db_mask &= np.array([not m.pano for m in self.db_metadata()])

            if subtask:
                query_mask &= np.array([getattr(s, subtask) for s in self.query_subtasks()])
                db_mask &= np.array([getattr(s, subtask) for s in self.db_subtasks()])

            self.query_mask, self.db_mask = query_mask, db_mask

        self.exclude_panos = exclude_panos
        self.subtask = subtask

    @property
    def sub_dir(self) -> str:
        if self.name in SCENES_TRAIN + SCENES_VAL:
            return "train_val"
        elif self.name in SCENES_TEST:
            return "test"
        else:
            raise RuntimeError(f"Invalid scene name {self.name}")

    @property
    def image_dir(self) -> Path:
        return self.dataset_dir / self.sub_dir / self.name

    def _read_csv(self, path: Path, mask: Optional[np.ndarray] = None) -> Tuple[str, List[List]]:
        with open(path) as f:
            header = f.readline().strip()
            data = [line.strip().split(",") for line in f]
        if mask is not None:
            assert mask.shape[0] == len(data)
            data = [d for d, m in zip(data, mask) if m]
        return header, data

    def _get_image_paths(self, path: Path, mask: Optional[np.ndarray] = None) -> List[Path]:
        base_dir = (path.parent / "images").relative_to(self.image_dir)
        header, data = self._read_csv(path, mask)
        assert header == ",key,sequence_key,frame_number"
        return [base_dir / f"{d[1]}.jpg" for d in data]

    def query_image_paths(self) -> List[Path]:
        return self._get_image_paths(self.image_dir / "query" / "seq_info.csv", self.query_mask)

    def db_image_paths(self) -> List[Path]:
        return self._get_image_paths(self.image_dir / "database" / "seq_info.csv", self.db_mask)

    def _get_utm_coords(self, path: Path, mask: Optional[np.ndarray] = None) -> np.ndarray:
        header, data = self._read_csv(path, mask)
        assert header == ",key,easting,northing,unique_cluster,control_panel,night,view_direction"
        return np.array([[float(d[2]), float(d[3])] for d in data])

    def query_utm_coords(self) -> np.ndarray:
        return self._get_utm_coords(self.image_dir / "query" / "postprocessed.csv", self.query_mask)

    def db_utm_coords(self) -> np.ndarray:
        return self._get_utm_coords(self.image_dir / "database" / "postprocessed.csv", self.db_mask)

    def _get_metadata(self, path: Path, mask: Optional[np.ndarray] = None) -> List[Metadata]:
        header, data = self._read_csv(path, mask)
        assert header == ",key,lon,lat,ca,captured_at,pano"
        return [Metadata(d[1], float(d[2]), float(d[3]), float(d[4]), d[5], d[6] == "True") for d in data]

    def query_metadata(self) -> List[Metadata]:
        return self._get_metadata(self.image_dir / "query" / "raw.csv", self.query_mask)

    def db_metadata(self) -> List[Metadata]:
        return self._get_metadata(self.image_dir / "database" / "raw.csv", self.db_mask)

    def _get_subtasks(self, path: Path, mask: Optional[np.ndarray]) -> List[Subtasks]:
        header, data = self._read_csv(path, mask)
        fields = header.split(",")
        field2idx = {f: idx for idx, f in enumerate(fields)}
        idx = [field2idx[f] for f in Subtasks._fields]
        return [Subtasks(*[d[i] == "True" for i in idx]) for d in data]

    def query_subtasks(self) -> List[Subtasks]:
        return self._get_subtasks(self.image_dir / "query" / "subtask_index.csv", self.query_mask)

    def db_subtasks(self) -> List[Subtasks]:
        return self._get_subtasks(self.image_dir / "database" / "subtask_index.csv", self.db_mask)
