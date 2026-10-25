COMP0227 Lab 04, Part 2: the labelled set of `object-detection --evaluate data/detection_sets/room0`.

boxes.json holds the 351 true boxes (chair 84, couch 60, dining table 107, vase 55, book 45) of 59
keyframes of the Replica room0 scene, in the COCO format. The boxes were made from the scene's per-pixel
class maps, so touching objects of one class share a box, and an object cut in two by something in front
of it has two.

The images are not here. They are frames of the Replica dataset (Straub et al., "The Replica Dataset: A
Digital Replica of Indoor Spaces", arXiv 1906.05797), rendered as in NICE-SLAM: the files
results/frameNNNNNN.jpg (1200 x 680) of the room0 folder of the Replica package which pySLAM's README
links. Give that folder with --images:

    pixi run -e semantics object-detection --method YOLO --evaluate data/detection_sets/room0 \
        --images path/to/Replica/room0

`pixi run -e semantics models-detection --images path/to/Replica/room0` checks that the 59 images are
there, and that they are the files the boxes were made on.
