import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from pycolmap import Reconstruction

from ..fov import fov2d_overlap_pairs
from ..utils import get_descriptors, root_dir
from .scene import LamarScene

RADIO_MAX_DIST_M = 500.0
RADIO_MAX_DELAY_S = 10.0
BT_FREQ_MHZ = 2.4e3
BETA = 2.5e-4


def signal_strength_to_dist(freq_mhz: float, rssi_dbm: float) -> float:
    return np.power(10.0, (27.55 - 20.0 * np.log10(freq_mhz) + abs(rssi_dbm)) / 20.0)


def radio_desc(
    image_wifi_list: List[List[LamarScene.Wifi]], image_bt_list: List[List[LamarScene.Bt]], radio_to_idx: Dict[str, int]
) -> torch.Tensor:
    start = time.time()
    feat = torch.full((len(image_wifi_list), len(radio_to_idx)), RADIO_MAX_DIST_M)

    for i, wifis in enumerate(image_wifi_list):
        for w in wifis:
            freq_mhz = 2.4e3 if w.frequency_khz == -1 else w.frequency_khz / 1e3
            feat[i, radio_to_idx[w.mac_addr]] = signal_strength_to_dist(freq_mhz, w.rssi_dbm)

    for i, bts in enumerate(image_bt_list):
        for b in bts:
            feat[i, radio_to_idx[b.id]] = signal_strength_to_dist(BT_FREQ_MHZ, b.rssi_dbm)

    feat = np.minimum(feat, RADIO_MAX_DIST_M)
    elapsed = time.time() - start
    logging.debug(f"Computed {feat.shape[0]} radio descriptors in {elapsed:.3f} s")
    return feat


def get_radio_desc(scene: LamarScene, query_from_db: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """! Compute radio descriptors for the database and query images.

    @param scene Scene.
    @param query_from_db Pick query images from the database.
    @return The radio descriptors.
    """
    db_image_wifi = scene.db_image_wifi(RADIO_MAX_DELAY_S)
    query_image_wifi = db_image_wifi if query_from_db else scene.query_image_wifi(RADIO_MAX_DELAY_S)

    db_image_bt = scene.db_image_bt(RADIO_MAX_DELAY_S)
    query_image_bt = db_image_bt if query_from_db else scene.query_image_bt(RADIO_MAX_DELAY_S)

    all_wifis = set([w.mac_addr for wifi_list in db_image_wifi + query_image_wifi for w in wifi_list])
    all_bts = set([b.id for bt_list in db_image_bt + query_image_bt for b in bt_list])
    all_radios = all_wifis.union(all_bts)
    radio_to_idx = {w: idx for idx, w in enumerate(all_radios)}

    db_radio_desc = radio_desc(db_image_wifi, db_image_bt, radio_to_idx)
    query_radio_desc = db_radio_desc if query_from_db else radio_desc(query_image_wifi, query_image_bt, radio_to_idx)
    return db_radio_desc, query_radio_desc


def image_poses(reconstruction: Reconstruction, image_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """! Compute image poses.

    @param reconstruction COLMAP reconstruction.
    @param image_names List of image names for which to get poses (N).
    @return The camera positions (N, 3) and the heading angles (N).
    """
    xyz = np.empty((len(image_names), 3))
    hdg = np.empty(len(image_names))

    images = reconstruction.images
    name2key = dict()
    for key, im in images.items():
        name2key[im.name] = key

    for idx, image_name in enumerate(image_names):
        image = images[name2key[image_name]]
        # See https://colmap.github.io/format.html#images-txt
        R, t = image.cam_from_world.rotation.matrix(), image.cam_from_world.translation
        xyz[idx] = -R.T @ t  # Global position
        hdg[idx] = np.arctan2(R[2, 1], R[2, 0])  # Angle in xy plane

    return xyz, hdg


class LamarDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene: LamarScene,
        desc_path: Path,
        N: int,
        query_from_db: bool,
        affinity: List[str],
        radio_dropout_p: float = 0.0,
    ) -> None:
        """! Class initializer.

        @param scene Scene.
        @param desc_path Path to HDF5 file with global image descriptors.
        @param N Number of database images to return for each query.
        @param query_from_db Pick query images from the database.
        @param affinity Types of affinity ("positional-db": 2D field-of-view overlap, "radio"/"radio-db": WiFi/Bluetooth)
        @param radio_dropout_p Radio dropout probability.
        """
        super().__init__()

        logging.debug(f"LaMAR scene {scene.name}")

        self.db_image_list = list(map(str, scene.db_image_paths()))
        query_image_list = self.db_image_list if query_from_db else list(map(str, scene.query_image_paths()))

        logging.debug(f"{len(self.db_image_list)} db images and {len(query_image_list)} query images")

        self.db_desc = torch.from_numpy(get_descriptors(desc_path, self.db_image_list))
        self.query_desc = (
            self.db_desc if query_from_db else torch.from_numpy(get_descriptors(desc_path, query_image_list))
        )

        path = root_dir() / "data" / f"lamar_scores_{scene.name.lower()}_{'db' if query_from_db else 'query'}.npz"
        data = np.load(path)
        assert N <= data["topk"].shape[1]
        self.topk = torch.from_numpy(data["topk"][:, :N]).int()
        self.labels = torch.from_numpy(data["labels"][:, :N])
        self.num_pos = None if query_from_db else torch.from_numpy(data["num_pos"])

        if "positional-db" in affinity:
            reconstruction = scene.reconstruction(include_query_images=False)
            self.db_coords, self.db_heading = image_poses(reconstruction, self.db_image_list)

        if "radio" in affinity or "radio-db" in affinity:
            self.db_radio_desc, self.query_radio_desc = get_radio_desc(scene, query_from_db)

        self.affinity = affinity
        self.radio_dropout_p = radio_dropout_p

    def _affinity(self, query_idx: int, db_idx: np.ndarray) -> List[torch.Tensor]:
        aff = []

        for a in self.affinity:
            if a == "positional-db":
                db_z = self.db_coords[db_idx, 2]
                mask = np.abs(db_z - db_z[:, None]) <= 3.0
                pos_aff = torch.from_numpy(
                    fov2d_overlap_pairs(self.db_coords[db_idx, :2], self.db_heading[db_idx], r=10.0, mask=mask)
                )
                aff.append(pos_aff)
            elif a in ("radio", "radio-db"):
                radio_desc = self.db_radio_desc[db_idx]
                if a == "radio":
                    radio_desc = torch.cat((self.query_radio_desc[query_idx].unsqueeze(0), radio_desc), dim=0)
                if self.radio_dropout_p > 0.0:
                    mask = radio_desc != RADIO_MAX_DIST_M
                    mask *= torch.rand(radio_desc.shape) < self.radio_dropout_p
                    radio_desc[mask] = RADIO_MAX_DIST_M
                dist = torch.cdist(radio_desc, radio_desc)
                aff.append(1.0 - BETA * dist)
            else:
                raise ValueError(f"Invalid affinity type {a}")

        return aff

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """! Get item.

        @param idx Query image index (into self.query_desc / self.topk).
        @return The image descriptors (1+N, D) where the query is at index 0, the thresholded Sampson score between
                query and the other images (1+N), the affinity between images ((N, N) or (1+N, 1+N)), the database
                image indices (N) and the number of relevant database images.
        """
        db_idx = self.topk[idx]
        query_desc = self.query_desc[idx].unsqueeze(0)
        db_desc = self.db_desc[db_idx]
        desc = torch.cat((query_desc, db_desc), dim=0)

        if self.labels is None:
            labels = []
        else:
            labels = torch.cat((torch.ones(1, dtype=bool), self.labels[idx]))

        aff = self._affinity(idx, db_idx)

        num_pos = [] if self.num_pos is None else self.num_pos[idx]

        return desc, labels, aff, db_idx, num_pos

    def __len__(self) -> int:
        """! Get number of items.

        @return The number of items in the dataset.
        """
        return self.query_desc.shape[0]
