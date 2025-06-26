import numpy as np
from scipy.spatial import ConvexHull

# Helper functions within the main function
def boxes_intersect(bbox1, bbox2):
    """Check if two boxes intersect"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to [xmin, ymin, xmax, ymax] format
    xmin1, ymin1, xmax1, ymax1 = x1, y1, x1 + w1, y1 + h1
    xmin2, ymin2, xmax2, ymax2 = x2, y2, x2 + w2, y2 + h2

    # No intersection if one box is to the left/right/above/below the other
    return not (xmax1 <= xmin2 or xmax2 <= xmin1 or ymax1 <= ymin2 or ymax2 <= ymin1)

def is_box_inside(bbox1, bbox2):
    """Check if bbox1 is completely inside bbox2"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to [xmin, ymin, xmax, ymax] format
    xmin1, ymin1, xmax1, ymax1 = x1, y1, x1 + w1, y1 + h1
    xmin2, ymin2, xmax2, ymax2 = x2, y2, x2 + w2, y2 + h2

    # Check if bbox1 is inside bbox2
    return xmin1 >= xmin2 and ymin1 >= ymin2 and xmax1 <= xmax2 and ymax1 <= ymax2

def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to [xmin, ymin, xmax, ymax] format
    xmin1, ymin1, xmax1, ymax1 = x1, y1, x1 + w1, y1 + h1
    xmin2, ymin2, xmax2, ymax2 = x2, y2, x2 + w2, y2 + h2

    # Calculate area of each box
    area1 = w1 * h1
    area2 = w2 * h2

    # Calculate coordinates of intersection
    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)

    # Check if there is an intersection
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0

    # Calculate area of intersection
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

    # Calculate IoU
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0

    return iou

def compute_convex_hull(boxes):
    """Compute convex hull of multiple bounding boxes"""
    all_corners = []

    # Collect all corner points from all boxes
    for bbox in boxes:
        x, y, w, h = bbox
        corners = [
            [x, y],  # Top-left
            [x + w, y],  # Top-right
            [x, y + h],  # Bottom-left
            [x + w, y + h],  # Bottom-right
        ]
        all_corners.extend(corners)

    all_corners = np.array(all_corners)

    # Compute convex hull
    hull = ConvexHull(all_corners)
    hull_points = all_corners[hull.vertices]

    # Find xmin, ymin, width, height from hull points
    xmin = np.min(hull_points[:, 0])
    ymin = np.min(hull_points[:, 1])
    xmax = np.max(hull_points[:, 0])
    ymax = np.max(hull_points[:, 1])

    # Return in [xmin, ymin, width, height] format
    width = xmax - xmin
    height = ymax - ymin
    return [xmin, ymin, width, height]

def dfs(node, graph, visited, component):
    """Depth-first search to find connected components"""
    visited[node] = True
    component.append(node)

    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs(neighbor, graph, visited, component)

def postprocess_bboxes(bboxes, process_intersecting=True, iou_threshold=0.3, remove_contained=True):
    """
    Process bounding boxes to merge overlapping ones and eliminate contained boxes.

    Args:
        bboxes: List of bounding boxes in format [xmin, ymin, width, height]
        process_intersecting: If True, process both contained and intersecting boxes.
                             If False, only process completely contained boxes.
        iou_threshold: Only process boxes with IoU greater than this threshold (0.0 to 1.0)
        remove_contained: If True, remove boxes that are completely inside other boxes
                         before any further processing

    Returns:
        List of processed bounding boxes in the same format
    """
    processed_boxes = []


    # Main processing logic
    n = len(bboxes)
    if n == 0:
        return []

    # First pass: Remove contained boxes if requested
    if remove_contained:
        # Track which boxes to keep
        boxes_to_keep = [True] * n

        # Check each pair of boxes
        for i in range(n):
            if not boxes_to_keep[i]:
                continue

            for j in range(n):
                if i == j or not boxes_to_keep[j]:
                    continue

                # If box j is inside box i, mark j to be removed
                if is_box_inside(bboxes[j], bboxes[i]):
                    boxes_to_keep[j] = False

        # Create new list with only the boxes to keep
        bboxes = [bbox for idx, bbox in enumerate(bboxes) if boxes_to_keep[idx]]

        # If all boxes were filtered or only one remains, return early
        if len(bboxes) <= 1:
            return bboxes

    # Update n after potential filtering
    n = len(bboxes)

    # Create a graph to track intersecting boxes
    graph = [[] for _ in range(n)]

    # Build intersection graph based on parameters and IoU threshold
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate IoU to check significant overlap
            iou = calculate_iou(bboxes[i], bboxes[j])

            # Check if boxes intersect or are contained
            boxes_overlap = boxes_intersect(bboxes[i], bboxes[j])
            box1_inside_box2 = is_box_inside(bboxes[i], bboxes[j])
            box2_inside_box1 = is_box_inside(bboxes[j], bboxes[i])

            # Determine if we should connect these boxes in the graph
            should_connect = False

            if iou >= iou_threshold:
                if process_intersecting:
                    # Process both intersecting and contained boxes
                    should_connect = boxes_overlap
                else:
                    # Only process contained boxes
                    should_connect = box1_inside_box2 or box2_inside_box1

            if should_connect:
                graph[i].append(j)
                graph[j].append(i)

    # Find connected components (groups of intersecting boxes)
    visited = [False] * n
    components = []

    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, graph, visited, component)
            components.append(component)

    # Process each connected component
    for component in components:
        # Handle single boxes directly
        if len(component) == 1:
            processed_boxes.append(bboxes[component[0]])
            continue

        # Handle each group of connected boxes
        component_boxes = [bboxes[i] for i in component]

        # Filter out boxes that are completely inside other boxes
        filtered_indices = set(range(len(component_boxes)))
        for i in range(len(component_boxes)):
            if i not in filtered_indices:
                continue

            for j in range(len(component_boxes)):
                if i == j or j not in filtered_indices:
                    continue

                if is_box_inside(component_boxes[i], component_boxes[j]):
                    # Box i is inside box j, ignore box i
                    filtered_indices.discard(i)
                    break

        filtered_boxes = [component_boxes[i] for i in filtered_indices]

        # Create final boxes based on filtering results
        if len(filtered_boxes) == 1:
            # Only one box left, keep it as is
            processed_boxes.append(filtered_boxes[0])
        elif len(filtered_boxes) > 1:
            # Multiple boxes, compute the convex hull
            merged_box = compute_convex_hull(filtered_boxes)
            processed_boxes.append(merged_box)

    return processed_boxes
