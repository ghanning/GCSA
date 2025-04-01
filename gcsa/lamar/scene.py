from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pycolmap import Camera, Image, Reconstruction, Rigid3d

SCENES = ["CAB", "HGE", "LIN"]


class LamarScene:
    """! LaMAR scene class.

    An instance of this class represents one scene of the dataset.
    """

    Query = namedtuple("Query", ["timestamp", "sensor_id"])
    Image = namedtuple("Image", ["timestamp", "sensor_id", "image_path"])
    Pose = namedtuple("Pose", ["timestamp", "device_id", "q", "t"])
    Sensor = namedtuple("Sensor", ["sensor_id", "name", "sensor_type", "sensor_params"])
    Rig = namedtuple("Rig", ["rig_id", "sensor_id", "q", "t"])
    Wifi = namedtuple(
        "Wifi",
        [
            "timestamp",
            "sensor_id",
            "mac_addr",
            "frequency_khz",
            "rssi_dbm",
            "name",
            "scan_time_start_us",
            "scan_time_end_u",
        ],
    )
    Bt = namedtuple("Bt", ["timestamp", "sensor_id", "id", "rssi_dbm", "name"])

    def __init__(
        self, name: str, dataset_dir: Path, devices: Tuple[str, ...] = ("hololens", "phone"), validation: bool = True
    ) -> None:
        """! Class initializer.

        @param name Scene name.
        @param dataset_dir Path to dataset root directory.
        @param devices Which devices to consider ("hololens" and/or "phone").
        @param validation Whether to use the validation queries.
        """
        self.name = name
        self.dataset_dir = dataset_dir
        for d in devices:
            if d not in ("phone", "hololens"):
                raise ValueError(f"Device {d} not supported")
        self.devices = devices
        self.session_dir = dataset_dir / self.name / "sessions"
        self.map_dir = self.session_dir / "map"
        self.validation = validation

    @property
    def image_dir(self) -> Path:
        return self.dataset_dir / self.name

    def _val_dir(self, device: str) -> Path:
        subdir = f"query_val_{device}" if self.validation else f"query_{device}"
        return self.session_dir / subdir

    def _read_csv(self, path: Path) -> List[List]:
        with open(path) as f:
            rows = [line.rstrip("\n").split(", ") for line in f if not line.startswith("#")]
        return rows

    def _read_query_list(self, device: str) -> List[LamarScene.Query]:
        rows = self._read_csv(self._val_dir(device) / "queries.txt")
        return [LamarScene.Query(int(r[0]), r[1]) for r in rows]

    def _read_image_list(self, path: Path) -> List[LamarScene.Image]:
        rows = self._read_csv(path)
        return [LamarScene.Image(int(r[0]), r[1], Path(r[2])) for r in rows]

    def _read_pose_list(self, path: Path) -> List[LamarScene.Pose]:
        rows = self._read_csv(path)
        return [
            LamarScene.Pose(int(r[0]), r[1], np.array(list(map(float, r[2:6]))), np.array(list(map(float, r[6:9]))))
            for r in rows
        ]

    def _read_sensor_list(self, path: Path) -> List[LamarScene.Sensor]:
        rows = self._read_csv(path)
        return [LamarScene.Sensor(r[0], r[1], r[2], r[3:]) for r in rows]

    def _read_rig_list(self, path: Path) -> List[LamarScene.Rig]:
        rows = self._read_csv(path)
        return [
            LamarScene.Rig(r[0], r[1], np.array(list(map(float, r[2:6]))), np.array(list(map(float, r[6:9]))))
            for r in rows
        ]

    def _read_wifi_list(self, path: Path) -> List[LamarScene.Wifi]:
        rows = self._read_csv(path)
        return [
            LamarScene.Wifi(int(r[0]), r[1], r[2], int(r[3]), float(r[4]), r[5], int(r[6]), int(r[7])) for r in rows
        ]

    def _read_bt_list(self, path: Path) -> List[LamarScene.Bt]:
        rows = self._read_csv(path)
        return [LamarScene.Bt(int(r[0]), r[1], r[2], float(r[3]), r[4]) for r in rows]

    def _query_image_list(self, device: str) -> List[LamarScene.Image]:
        val_dir = self._val_dir(device)
        query_list = self._read_query_list(device)
        image_list = self._read_image_list(val_dir / "images.txt")
        sensor_to_path = {(i.timestamp, i.sensor_id): i.image_path for i in image_list}
        assert len(image_list) == len(sensor_to_path)

        base_dir = val_dir / "raw_data"

        if device == "phone":
            return [
                LamarScene.Image(
                    q.timestamp,
                    q.sensor_id,
                    (base_dir / sensor_to_path[(q.timestamp, q.sensor_id)]).relative_to(self.image_dir),
                )
                for q in query_list
            ]
        else:  # if device == "hololens"
            rig_list = self._read_rig_list(val_dir / "rigs.txt")
            rig_to_sensors = {}
            for rig in rig_list:
                if rig.rig_id not in rig_to_sensors:
                    rig_to_sensors[rig.rig_id] = list()
                rig_to_sensors[rig.rig_id].append(rig.sensor_id)

            return [
                LamarScene.Image(
                    q.timestamp,
                    sensor_id,
                    (base_dir / sensor_to_path[(q.timestamp, sensor_id)]).relative_to(self.image_dir),
                )
                for q in query_list
                for sensor_id in rig_to_sensors[q.sensor_id]
            ]

    def _db_image_list(self) -> List[LamarScene.Image]:
        image_list = self._read_image_list(self.map_dir / "images.txt")
        return [
            LamarScene.Image(
                i.timestamp, i.sensor_id, (self.map_dir / "raw_data" / i.image_path).relative_to(self.image_dir)
            )
            for i in image_list
        ]

    def _add_cameras(
        self, reconstruction: Reconstruction, sensor_list: List[LamarScene.Sensor], sensor_to_cam: Dict[str, int]
    ):
        for sensor in sensor_list:
            if sensor.sensor_type != "camera":
                continue
            model = sensor.sensor_params[0]
            width = int(sensor.sensor_params[1])
            height = int(sensor.sensor_params[2])
            params = list(map(float, sensor.sensor_params[3:]))
            id = len(reconstruction.cameras) + 1
            camera = Camera(model=model, width=width, height=height, params=params, camera_id=id)
            assert sensor.sensor_id not in sensor_to_cam
            sensor_to_cam[sensor.sensor_id] = id
            reconstruction.add_camera(camera)

    def _add_images(
        self,
        reconstruction: Reconstruction,
        image_list: List[LamarScene.Image],
        pose_list: List[LamarScene.Pose],
        rig_list: List[LamarScene.Rig],
        sensor_to_cam: Dict[str, int],
        absolute_paths: bool,
    ):
        device_to_pose = {(p.timestamp, p.device_id): (p.q, p.t) for p in pose_list}
        if rig_list:
            sensor_to_rigs = dict()
            for rig in rig_list:
                if rig.sensor_id not in sensor_to_rigs:
                    sensor_to_rigs[rig.sensor_id] = list()
                sensor_to_rigs[rig.sensor_id].append(rig)

        for image in image_list:
            if (image.timestamp, image.sensor_id) in device_to_pose:
                q, t = device_to_pose[(image.timestamp, image.sensor_id)]
                T = Rigid3d(q, t)
            else:
                rigs = sensor_to_rigs[image.sensor_id]
                rig = next(r for r in rigs if (image.timestamp, r.rig_id) in device_to_pose)
                q1, t1 = rig.q, rig.t  # sensor -> rig
                T1 = Rigid3d(q1, t1)
                q2, t2 = device_to_pose[(image.timestamp, rig.rig_id)]  # rig -> world
                T2 = Rigid3d(q2, t2)
                T = T2 * T1

            # COLMAP uses world-to-camera, poses given in LaMAR are camera-to-world
            name = str((self.image_dir / image.image_path).absolute() if absolute_paths else image.image_path)
            id = len(reconstruction.images) + 1
            image = Image(name=name, cam_from_world=T.inverse(), camera_id=sensor_to_cam[image.sensor_id], image_id=id)
            reconstruction.add_image(image)

    def reconstruction(self, include_query_images: bool = True, absolute_paths: bool = False) -> Reconstruction:
        reconstruction = Reconstruction()
        sensor_to_cam = dict()

        # Cameras
        sensor_list = self._read_sensor_list(self.map_dir / "sensors.txt")
        self._add_cameras(reconstruction, sensor_list, sensor_to_cam)

        # Images
        image_list = self._db_image_list()
        pose_list = self._read_pose_list(self.map_dir / "trajectories.txt")
        rig_list = self._read_rig_list(self.map_dir / "rigs.txt")
        self._add_images(reconstruction, image_list, pose_list, rig_list, sensor_to_cam, absolute_paths)

        if include_query_images:
            for device in self.devices:
                val_dir = self._val_dir(device)

                # Cameras
                # TODO: Exclude cameras not in query list?
                sensor_list = self._read_sensor_list(val_dir / "sensors.txt")
                self._add_cameras(reconstruction, sensor_list, sensor_to_cam)

                # Images
                image_list = self._query_image_list(device)
                pose_list = self._read_pose_list(val_dir / "proc" / "alignment_trajectories.txt")
                rig_list = None if device == "phone" else self._read_rig_list(val_dir / "rigs.txt")
                self._add_images(reconstruction, image_list, pose_list, rig_list, sensor_to_cam, absolute_paths)

        return reconstruction

    def query_image_paths(self) -> List[Path]:
        return [i.image_path for d in self.devices for i in self._query_image_list(d)]

    def db_image_paths(self) -> List[Path]:
        image_list = self._db_image_list()
        return [i.image_path for i in image_list]

    def write_query_list(self, path: Path):
        with open(path, "w") as f:
            for device in self.devices:
                sensor_list = self._read_sensor_list(self._val_dir(device) / "sensors.txt")
                id_to_sensor = {s.sensor_id: s for s in sensor_list}

                for image in self._query_image_list(device):
                    sensor = id_to_sensor[image.sensor_id]
                    f.write(f"{image.image_path} {' '.join(sensor.sensor_params)}\n")

    def session_name(self, image_path: Path) -> str:
        # Get session from image path
        parents = image_path.parents
        for i in range(len(parents)):
            if parents[i].name == "raw_data":
                return parents[i - 1].name
        raise ValueError(f"Path {image_path} does not contain 'raw_data'")

    def device_name(self, image_path: Path) -> str:
        session = self.session_name(image_path)
        if session.startswith("hl"):
            return "hololens"
        elif session.startswith("ios"):
            return "phone"
        else:
            raise ValueError(f"Cannot determine device for {image_path}")

    def _image_wifi(
        self, wifi_list: List[LamarScene.Wifi], image_list: List[LamarScene.Image], max_delay_s: float, future: bool
    ) -> List[List[LamarScene.Wifi]]:
        # Group measurements by sensor and timestamp
        id_to_wifi = dict()
        for wifi in wifi_list:
            id = wifi.sensor_id.split("/")[0]
            if id not in id_to_wifi:
                id_to_wifi[id] = dict()
            if wifi.timestamp not in id_to_wifi[id]:
                id_to_wifi[id][wifi.timestamp] = list()
            id_to_wifi[id][wifi.timestamp].append(wifi)

        # For each image find the closest group of measurements in time
        image_wifi_list = list()
        for image in image_list:
            id = image.sensor_id.split("/")[0]
            wifis = []
            if id in id_to_wifi:
                dt_min, timestamp_min = None, None
                for timestamp in id_to_wifi[id].keys():
                    if not future and timestamp > image.timestamp:
                        continue
                    dt = abs(image.timestamp - timestamp)
                    if dt_min is None or dt < dt_min:
                        dt_min = dt
                        timestamp_min = timestamp
                if dt_min is not None and dt_min <= max_delay_s * 1e6:
                    wifis = id_to_wifi[id][timestamp_min]
            image_wifi_list.append(wifis)

        return image_wifi_list

    def query_image_wifi(self, max_delay_s: float) -> List[List[LamarScene.Wifi]]:
        image_wifi_list = list()

        for device in self.devices:
            wifi_list = self._read_wifi_list(self._val_dir(device) / "wifi.txt")
            image_list = self._query_image_list(device)
            image_wifi_list.extend(self._image_wifi(wifi_list, image_list, max_delay_s, False))

        return image_wifi_list

    def db_image_wifi(self, max_delay_s: float) -> List[List[LamarScene.Wifi]]:
        wifi_list = self._read_wifi_list(self.map_dir / "wifi.txt")
        image_list = self._db_image_list()
        return self._image_wifi(wifi_list, image_list, max_delay_s, True)

    def _image_bt(
        self, bt_list: List[LamarScene.Bt], image_list: List[LamarScene.Image], max_delay_s: float, future: bool
    ) -> List[List[LamarScene.Bt]]:
        # Group measurements by sensor
        id_to_bt = dict()
        for bt in bt_list:
            id = bt.sensor_id.split("/")[0]
            if id not in id_to_bt:
                id_to_bt[id] = list()
            id_to_bt[id].append(bt)

        # For each image find the closest measurements in time
        image_bt_list = list()
        for image in image_list:
            id = image.sensor_id.split("/")[0]
            bts = dict()
            if id in id_to_bt:
                for bt in id_to_bt[id]:
                    if not future and bt.timestamp > image.timestamp:
                        continue
                    dt = abs(image.timestamp - bt.timestamp)
                    if dt > max_delay_s * 1e6:
                        continue
                    if bt.id not in bts or dt < abs(image.timestamp - bts[bt.id].timestamp):
                        bts[bt.id] = bt
            image_bt_list.append(bts.values())

        return image_bt_list

    def query_image_bt(self, max_delay_s: float) -> List[List[LamarScene.Bt]]:
        image_bt_list = list()

        for device in self.devices:
            bt_list = self._read_bt_list(self._val_dir(device) / "bt.txt")
            image_list = self._query_image_list(device)
            image_bt_list.extend(self._image_bt(bt_list, image_list, max_delay_s, False))

        return image_bt_list

    def db_image_bt(self, max_delay_s: float) -> List[List[LamarScene.Bt]]:
        bt_list = self._read_bt_list(self.map_dir / "bt.txt")
        image_list = self._db_image_list()
        return self._image_bt(bt_list, image_list, max_delay_s, True)
