"""
The messages between pySLAM and its workers, over a multiprocessing.connection Connection.

A message is a small JSON header followed by the raw bytes of its arrays, one frame each. The header
holds the message (dicts, lists, numbers, strings) with every numpy array and every list of cv2.KeyPoint
replaced by a reference to one of the frames, plus each array's dtype and shape. Nothing is pickled, so
both sides can use different Python packages, and the same messages can go to a remote machine later.
"""

import json

import cv2
import numpy as np

# cv2.KeyPoint as one row: x, y, size, angle, response, octave, class_id
KEYPOINT_FIELDS = 7


def _keypoints_to_array(kps):
    a = np.empty((len(kps), KEYPOINT_FIELDS), dtype=np.float32)
    for i, kp in enumerate(kps):
        a[i] = (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
    return a


def _array_to_keypoints(a):
    return [
        cv2.KeyPoint(
            x=float(r[0]), y=float(r[1]), size=float(r[2]), angle=float(r[3]),
            response=float(r[4]), octave=int(r[5]), class_id=int(r[6]),
        )
        for r in a
    ]


def _is_keypoint_list(obj):
    return isinstance(obj, (list, tuple)) and len(obj) > 0 and all(isinstance(k, cv2.KeyPoint) for k in obj)


def _encode(obj, arrays):
    if isinstance(obj, np.ndarray):
        arrays.append(obj)
        return {"__nd__": len(arrays) - 1}
    if _is_keypoint_list(obj):
        arrays.append(_keypoints_to_array(obj))
        return {"__kps__": len(arrays) - 1}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(o, arrays) for o in obj]}
    if isinstance(obj, list):
        return [_encode(o, arrays) for o in obj]
    if isinstance(obj, dict):
        return {str(k): _encode(v, arrays) for k, v in obj.items()}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"workers.protocol: cannot send a {type(obj).__name__}")


def _decode(obj, arrays):
    if isinstance(obj, dict):
        if "__nd__" in obj:
            return arrays[obj["__nd__"]]
        if "__kps__" in obj:
            return _array_to_keypoints(arrays[obj["__kps__"]])
        if "__tuple__" in obj:
            return tuple(_decode(o, arrays) for o in obj["__tuple__"])
        return {k: _decode(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(o, arrays) for o in obj]
    return obj


def send(conn, msg):
    arrays = []
    header = {"msg": _encode(msg, arrays), "arrays": []}
    for i, a in enumerate(arrays):
        a = np.ascontiguousarray(a)
        arrays[i] = a
        header["arrays"].append({"dtype": a.dtype.str, "shape": list(a.shape)})
    conn.send_bytes(json.dumps(header).encode("utf-8"))
    for a in arrays:
        conn.send_bytes(memoryview(a).cast("B") if a.size else b"")


def recv(conn):
    header = json.loads(conn.recv_bytes().decode("utf-8"))
    arrays = []
    for meta in header["arrays"]:
        data = conn.recv_bytes()
        arrays.append(np.frombuffer(data, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"]).copy())
    return _decode(header["msg"], arrays)
