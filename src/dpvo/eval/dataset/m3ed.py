import h5py
import numpy as np

from .abc import CameraType, Dataset


class M3EDDataset(Dataset):
    """M3ED dataset class."""

    def __init__(self, h5path, camera_name="/ovc/rgb", *args, **kwargs):
        """Initialize the M3ED dataset.

        Args:
            h5path (str): Path to the h5 file.
            camera (str): Camera dataset name.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.h5path = h5path
        self.camera_name = camera_name
        self.camera_device = "/".join(camera_name.split("/")[:-1])

        # Load the h5 file
        self.__load_h5()

        camera_model = self.h5.get(f"{camera_name}/calib/camera_model")[()]
        if camera_model.decode() != "pinhole":
            raise ValueError(
                f"Camera model {camera_model.decode()} is not supported. Only 'pinhole' is supported."
            )
        distortion_model = self.h5.get(f"{camera_name}/calib/distortion_model")[()]
        if distortion_model.decode() != "radtan":
            raise ValueError(
                f"Distortion model {distortion_model.decode()} is not supported. Only 'radtan' is supported."
            )

        h5_camera = self.h5[self.camera_name]
        self.__distortion = h5_camera["calib/distortion_coeffs"][()]
        self.__intrinsics_raw = h5_camera["calib/intrinsics"][()]
        self.__resolution_raw = tuple(h5_camera["calib/resolution"][()])
        self.__camera_model = CameraType.RADTAN
        self.__length_frames = self.h5_cam_data.shape[0]

        super().__init__(*args, **kwargs)

    def __load_h5(self):
        """Load the h5 file."""
        self.h5 = h5py.File(self.h5path, "r")
        self.h5_cam_data = self.h5[self.camera_name]["data"]
        self.h5_cam_ts = self.h5[self.camera_device]["ts"]
        if self.h5_cam_data.shape[0] != self.h5_cam_ts.shape[0]:
            raise ValueError(
                f"Camera data and timestamps do not match: {self.h5_cam_data.shape[0]} != {self.h5_cam_ts.shape[0]}"
            )

    @property
    def _distortion(self) -> np.ndarray:
        return self.__distortion

    @property
    def _intrinsics_raw(self) -> np.ndarray:
        return self.__intrinsics_raw

    @property
    def _camera_model(self) -> CameraType:
        return self.__camera_model

    @property
    def _resolution_raw(self) -> tuple:
        return self.__resolution_raw

    @property
    def length_frames(self) -> int:
        return self.__length_frames

    def _get_frame_raw(self, index: int) -> tuple:
        t = self.h5_cam_ts[index] / 1e6
        image = self.h5_cam_data[index]
        return t, image

    # Customize pickle to avoid pickling the h5py file
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["h5"], state["h5_cam_data"], state["h5_cam_ts"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.h5 = h5py.File(self.h5path, "r")
        self.h5_cam_data = self.h5[self.camera_name]["data"]
        self.h5_cam_ts = self.h5[self.camera_device]["ts"]
        if self.h5_cam_data.shape[0] != self.h5_cam_ts.shape[0]:
            raise ValueError(
                f"Camera data and timestamps do not match: {self.h5_cam_data.shape[0]} != {self.h5_cam_ts.shape[0]}"
            )


if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description="M3ED Dataset")
    parser.add_argument("--h5path", type=str, required=True, help="Path to the h5 file")
    parser.add_argument(
        "--camera", type=str, default="/ovc/rgb", help="Camera dataset name"
    )
    args = parser.parse_args()

    dataset = M3EDDataset(args.h5path, args.camera, clahe=True)
    print("Distortion:", dataset._distortion)
    print("Intrinsics:", dataset._intrinsics_raw)

    for t, image in dataset.iter_frames():
        print(f"Timestamp: {t}, Image shape: {image.shape}")
        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break