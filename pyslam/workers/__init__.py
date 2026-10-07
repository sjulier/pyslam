"""
Workers: model code that runs in another pixi environment, used by pySLAM through a socket.

The TensorFlow-based features (DELF, LF-Net, ContextDesc, GeoDesc) and the HDC-DELF place recognition
need TensorFlow, whose versions of protobuf and friends do not fit pySLAM's environments. They run in the
pixi environment `tf`, in a worker process (tf_worker.py) that pySLAM starts and talks to (tf_client.py).
"""
