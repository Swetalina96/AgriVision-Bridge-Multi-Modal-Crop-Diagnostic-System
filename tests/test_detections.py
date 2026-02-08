from pages.object_detection import extract_detections


def test_extract_detections_iterable_boxes():
    # Create fake box objects with cls and conf as sequences
    class Box:
        def __init__(self, cls, conf):
            self.cls = [cls]
            self.conf = [conf]

    class Boxes:
        def __init__(self, boxes):
            self._boxes = boxes
        def __iter__(self):
            return iter(self._boxes)

    class FakeRes:
        def __init__(self, boxes, names):
            self.boxes = boxes
            self.names = names

    boxes = Boxes([Box(0, 0.2467)])
    res = FakeRes(boxes, {0: 'Tomato___Early_blight'})
    dets = extract_detections(res)
    assert isinstance(dets, list)
    assert dets[0][0] == 'Tomato___Early_blight'
    # Allow small float tolerance
    assert abs(dets[0][1] - 0.2467) < 1e-6
