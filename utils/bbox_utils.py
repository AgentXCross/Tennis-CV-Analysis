def get_center_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    c = a * a + b * b
    return c ** 0.5