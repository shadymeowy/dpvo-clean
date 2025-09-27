import pathlib

import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm


def bag_image_iterator(
    path_bag,
    cam_topic,
    typestore=Stores.ROS1_NOETIC,
):
    typestore = get_typestore(Stores.ROS1_NOETIC)
    # Check if the bag file exists
    path_bag = pathlib.Path(path_bag)
    if not path_bag.is_file():
        raise FileNotFoundError(f"Bag file not found: {path_bag}")

    with AnyReader([path_bag], default_typestore=typestore) as reader:
        # Extract camera data from the bag file
        connections = [x for x in reader.connections if x.topic == cam_topic]
        for connection, _, rawdata in tqdm(
            reader.messages(connections=connections),
            desc=f"Processing {cam_topic}",
            total=len(connections),
        ):
            # Deserialize the message
            msg = reader.deserialize(rawdata, connection.msgtype)
            stamp = msg.header.stamp

            # Convert ROS timestamp to nanoseconds
            ts = (int(1e9) * stamp.sec + stamp.nanosec) / 1e9

            # Convert the ROS message to an image and save it
            img = msg_to_image(msg)

            # Yield the timestamp and image
            yield ts, img


def sync_generators(gen1, gen2, t_delta=0.01):
    try:
        t1, *data1 = next(gen1)
        t2, *data2 = next(gen2)
        while True:
            if abs(t1 - t2) < t_delta:
                yield (t1, *data1), (t2, *data2)
                t1, *data1 = next(gen1)
                t2, *data2 = next(gen2)
            elif t1 < t2:
                print(f"Skip frame at {t1} from first cam")
                t1, *data1 = next(gen1)
            else:
                print(f"Skip frame at {t2} from second cam")
                t2, *data2 = next(gen2)
    except StopIteration:
        return


def msg_to_image(msg):
    width = msg.width
    height = msg.height
    encoding = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)
    assert msg.step == width
    if encoding == "mono8":
        assert len(data) == width * height, (
            f"Data length {len(data)} does not match expected size {width * height} "
            f"for mono8 encoding."
        )
        img = data.reshape((height, width))
    elif encoding == "rgb8":
        assert len(data) == 3 * width * height, (
            f"Data length {len(data)} does not match expected size {3 * width * height} "
            f"for rgb8 encoding."
        )
        img = data.reshape((height, width, 3))
    elif encoding == "bgr8":
        assert len(data) == 3 * width * height, (
            f"Data length {len(data)} does not match expected size {3 * width * height} "
            f"for bgr3 encoding."
        )
        img = data.reshape((height, width, 3))
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")

    img = np.ascontiguousarray(img)
    return img


def read_calibration(path_conf):
    dist_order = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    intr_order = ["fx", "fy", "cx", "cy"]
    extr_order = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    res_order = ["width", "height"]

    def dict_to(d, order):
        dist = []
        for i, v in enumerate(order):
            if v in d:
                dist.append(d[v])
            else:
                dist.append(None)
        for v in reversed(dist):
            if v is not None:
                break
            dist.pop(-1)
        dist = np.array(dist)
        return dist

    with open(path_conf) as stream:
        conf = yaml.load(stream, yaml.FullLoader)
        res1 = dict_to(conf["cam1"], res_order)
        res2 = dict_to(conf["cam2"], res_order)
        intr1 = dict_to(conf["cam1"], intr_order)
        intr2 = dict_to(conf["cam2"], intr_order)
        dist1 = dict_to(conf["cam1"], dist_order)
        dist2 = dict_to(conf["cam2"], dist_order)
        extr = dict_to(conf["cam2_to_cam1"], extr_order)
    return res1, res2, intr1, intr2, dist1, dist2, extr
