import multiprocessing as mp
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Iterator, Tuple

import cv2
import numpy as np


class CameraType(Enum):
    """Enum for camera types."""

    UNKNOWN = 0
    RADTAN = 1
    EQUADISTANT = 2


class Dataset(ABC):
    def __init__(self, clahe=False):
        # Calculate undistort maps
        self.resolution = None
        self.intrinsics = None
        self._init_undistort()

        # Clahe
        if clahe:
            self._clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
        else:
            self._clahe = None

    @property
    @abstractmethod
    def length_frames(self) -> int:
        """Return the length of the frame sequence."""
        pass

    @property
    @abstractmethod
    def _distortion(self) -> np.ndarray:
        """Return the distortion parameters of the camera."""
        pass

    @property
    @abstractmethod
    def _intrinsics_raw(self) -> np.ndarray:
        """Return the intrinsic parameters of the camera before any processing."""
        pass

    @property
    @abstractmethod
    def _resolution_raw(self) -> Tuple[int, int]:
        """Return the resolution of the camera before any processing."""
        pass

    @abstractmethod
    def _get_frame_raw(self, index: int) -> Tuple[float, np.ndarray]:
        """Return the frame and its time at the given index before any processing."""
        pass

    @property
    @abstractmethod
    def _camera_model(self) -> CameraType:
        """Return the camera model."""
        pass

    def _init_undistort(self):
        """Initialize the undistort maps."""

        if self._camera_model == CameraType.RADTAN:
            # For now, target same resolution as input
            self.resolution = self._resolution_raw

            # First getOptimalNewCameraMatrix
            K = np.array(
                [
                    [self._intrinsics_raw[0], 0, self._intrinsics_raw[2]],
                    [0, self._intrinsics_raw[1], self._intrinsics_raw[3]],
                    [0, 0, 1],
                ]
            )

            # Get new camera matrix
            K_new, roi = cv2.getOptimalNewCameraMatrix(
                K,
                self._distortion,
                self._resolution_raw,
                0,
                self.resolution,
            )
            # Set the new intrinsics
            self.intrinsics = np.array(
                [
                    K_new[0, 0],
                    K_new[1, 1],
                    K_new[0, 2],
                    K_new[1, 2],
                ]
            )
            # Will be used to crop the frame later on
            self._roi = roi
            # Get the undistort maps
            self._mapx, self._mapy = cv2.initUndistortRectifyMap(
                K,
                self._distortion,
                None,
                K_new,
                self._resolution_raw,
                cv2.CV_32FC1,
            )
        elif self._camera_model == CameraType.EQUADISTANT:
            raise NotImplementedError(
                "Undistort maps for equidistant camera model not implemented yet"
            )
        else:
            raise NotImplementedError(
                f"Camera model {self._camera_model} not implemented yet"
            )

    def _undistort(self, frame: Any) -> Any:
        """Undistort the image using the undistort maps."""
        if self._camera_model == CameraType.RADTAN:
            # Undistort the frame
            frame = cv2.remap(
                frame,
                self._mapx,
                self._mapy,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            # Crop the frame
            x, y, w, h = self._roi
            frame = frame[y : y + h + 1, x : x + w + 1]

            # The frame is now in the new resolution
            assert frame.shape[0] == self.resolution[1]
            assert frame.shape[1] == self.resolution[0]

        return frame

    def _apply_clahe(self, frame: Any) -> Any:
        if not self._clahe:
            return frame

        if len(frame.shape) == 2:
            # Apply CLAHE to grayscale image
            frame = self._clahe.apply(frame)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            # To YUV color image
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = self._clahe.apply(yuv[:, :, 0])
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        elif len(frame.shape) == 3:
            # Apply CLAHE to each channel of the frame
            for i in range(frame.shape[2]):
                frame[:, :, i] = self._clahe.apply(frame[:, :, i])
        else:
            raise ValueError("Unsupported frame shape for CLAHE application")

        return frame

    def get_frame(self, index: int) -> Any:
        """Return the frame and its time at the given index after processing."""
        t, frame = self._get_frame_raw(index)

        # Check the frame shape
        assert frame.shape[0] == self.resolution[1]
        assert frame.shape[1] == self.resolution[0]

        frame = self._undistort(frame)
        frame = self._apply_clahe(frame)
        return t, frame

    def iter_frames(self) -> Iterator[Tuple[int, Any]]:
        """Iterate over the frames of the dataset."""
        for i in range(self.length_frames):
            yield self.get_frame(i)

    def iter_frames_mp(
        self, num_processes: int = mp.cpu_count()
    ) -> Iterator[Tuple[int, Any]]:
        """Iterate over the frames of the dataset using multiprocessing."""
        with mp.Pool(num_processes) as pool:
            for t, frame in pool.imap(self.get_frame, range(self.length_frames)):
                yield t, frame
