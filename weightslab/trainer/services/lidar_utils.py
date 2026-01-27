import logging
import os
import math
import numpy as np
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

def load_labels_for_scan(bin_path):
    """
    Given a path like .../velodyne/000001.bin, try to find .../label_velodyne/000001.txt
    Returns list of dicts: {'cls': str, 'box': [x, y, z, l, w, h, yaw]}
    """
    try:
        # Resolve Label Path
        # bin_path: /path/to/velodyne/xxxxx.bin
        # expected: /path/to/label_velodyne/xxxxx.txt
        dirname = os.path.dirname(bin_path)
        filename = os.path.basename(bin_path)
        file_id = os.path.splitext(filename)[0]
        
        # Check standard folder structure
        # ../velodyne -> ../label_velodyne
        parent = os.path.dirname(dirname)
        label_dir = os.path.join(parent, "label_velodyne")
        label_path = os.path.join(label_dir, f"{file_id}.txt")
        
        if not os.path.exists(label_path):
            return []

        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if not parts: continue
                
                # Format: Class x y z l w h heading
                cls_name = parts[0]
                if cls_name == "DontCare": continue
                
                # Parse floats
                vals = [float(x) for x in parts[1:]]
                if len(vals) < 7: continue
                
                labels.append({
                    'cls': cls_name,
                    'box': np.array(vals) # [x, y, z, l, w, h, yaw]
                })
        return labels
    except Exception as e:
        logger.warning(f"Failed to load labels for {bin_path}: {e}")
        return []

def get_corners_bev(box):
    """
    Get 4 corners of the bounding box in BEV (X-Y plane).
    box: [x, y, z, l, w, h, yaw]
    """
    x, y, z, l, w, h, yaw = box
    
    # Rotation matrix
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    
    # Corners in local coords (centered at 0)
    # l is along x, w is along y
    dx = l / 2
    dy = w / 2
    
    # Counter-clockwise 
    corners_local = np.array([
        [dx, dy],
        [-dx, dy],
        [-dx, -dy],
        [dx, -dy]
    ])
    
    # Rotate and translate
    corners_global = (R @ corners_local.T).T + np.array([x, y])
    return corners_global

def render_lidar(points, config, file_path=None):
    """
    Main entry point for LiDAR visualization. Dispatch options driven by config.
    """
    mode = config.get("mode", "bev")
    cmap_name = config.get("cmap", "turbo")
    max_range = config.get("max_range", 50.0)
    
    labels = []
    draw_labels = config.get("draw_labels", False)
    if file_path and draw_labels:
        labels = load_labels_for_scan(file_path)

    if mode == "range":
        rv_conf = config.get("range_view", {})
        return render_range_view(
            points, 
            width=rv_conf.get("width", 1024), 
            height=rv_conf.get("height", 64),
            fov_up=rv_conf.get("fov_up", 45.0),
            fov_down=rv_conf.get("fov_down", 45.0),
            stretch=rv_conf.get("vertical_stretch", 6),
            max_dist=max_range,
            cmap_name=cmap_name
        )
    elif mode in ["bev", "slice"]:
        bev_conf = config.get("bev", {})
        
        # SLICE Logic: Filter points by Z before rendering
        if mode == "slice":
            slice_conf = config.get("slice", {})
            z_min = slice_conf.get("z_min", -1.5)
            z_max = slice_conf.get("z_max", 1.0)
            
            # Filter
            z = points[:, 2]
            mask = (z >= z_min) & (z <= z_max)
            points_to_render = points[mask]
        else:
            points_to_render = points
        
        return render_bev(
            points_to_render,
            res=bev_conf.get("resolution", 0.1),
            size=bev_conf.get("image_size", 800),
            cx=bev_conf.get("center_x", 400),
            cy=bev_conf.get("center_y", 400),
            max_dist=max_range,
            cmap_name=cmap_name,
            labels=labels,
            brightness=bev_conf.get("brightness", 1.0)
        )


def render_bev(points, res=0.1, size=800, cx=400, cy=400, max_dist=50.0, cmap_name="turbo", labels=None, brightness=1.0):
    """
    Render Top-Down Bird's Eye View with optional Labels.
    """
    if points.size == 0:
        return Image.new("RGB", (size, size))
        
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2] # Use Z for color

    # 1. Map to pixels
    # Standard KITTI BEV:
    # X axis (forward) -> Image Up (-v)
    # Y axis (left) -> Image Left (-u)  OR  Y axis (left) -> Image Right (u)?
    # Usually we want X pointing UP in the image.
    # Image (0,0) is Top-Left. 
    # v = cy - x / res
    # u = cx - y / res  (if Y is Left positive)
    
    # Try typical setup:
    # X (Forward) -> Up (Negative V)
    # Y (Left) -> Left (Negative U)
    
    u = (cx - y / res).astype(np.int32)
    v = (cy - x / res).astype(np.int32)
    
    # 2. Filter out-of-bounds
    mask = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    
    u = u[mask]
    v = v[mask]
    z = z[mask]
    
    # 3. Canvas
    # Initialize black
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    
    if len(z) > 0:
        # 4. Color by Height (Z)
        # Normalize Z for colormap (-2m to +3m is typical usable range)
        z_min, z_max = -2.5, 3.0
        norm_z = np.clip((z - z_min) / (z_max - z_min), 0, 1)
        
        try:
             cmap = cm.get_cmap(cmap_name)
             
             # Apply Cmap
             colors = cmap(norm_z) # (N, 4)
             rgb = (colors[:, :3] * 255 * brightness).astype(np.uint8)
             canvas[v, u] = rgb
             
        except ImportError:
             # Fallback white
             canvas[v, u] = 255
    
    # Convert to PIL for drawing boxes
    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)

    # 5. Draw Labels
    if labels:
        for lbl in labels:
            cls = lbl['cls']
            box = lbl['box'] # x, y, z, l, w, h, yaw
            
            # Get corners
            corners = get_corners_bev(box)
            
            # Project to pixels
            poly_pts = []
            for cp in corners:
                cu = int(cx + cp[0] / res)
                cv = int(cy - cp[1] / res)
                poly_pts.append((cu, cv))
            
            # Color based on class?
            color = "green"
            if cls == "Car": color = "cyan"
            elif cls == "Pedestrian": color = "red"
            elif cls == "Cyclist": color = "yellow"
            
            # Draw Polygon
            draw.polygon(poly_pts, outline=color, width=3) # Increased width from default 1 to 3
            
            # Draw Heading Line (Front of box)
            # Front is traditionally +X in Kitti Label/Box convention here
            # Corners are: FR, FL, BL, BR?
            # get_corners_bev: [dx, dy] is Front-Left? No.
            # 0: [dx, dy], 1: [-dx, dy]...
            # dx is forward (l/2).
            # So 0-3 is front face? No. 0 is (+,+). 3 is (+,-)
            # Midpoint of 0 and 3
            front_u = (poly_pts[0][0] + poly_pts[3][0]) / 2
            front_v = (poly_pts[0][1] + poly_pts[3][1]) / 2
            draw.line([poly_pts[0], poly_pts[3]], fill="white", width=3) # Increased heading line width too
            
    return img

def render_bev_mask(labels, res=0.1, size=800, cx=400, cy=400, class_map=None):
    """
    Render Top-Down Bird's Eye View Label Mask (H, W).
    """
    if class_map is None:
        class_map = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}

    # Initialize black (background = 0)
    canvas = np.zeros((size, size), dtype=np.uint8)
    
    # Use PIL for polygon drawing (easier/faster than rolling our own rasterizer)
    img = Image.fromarray(canvas, mode="L")
    draw = ImageDraw.Draw(img)

    if labels:
        for lbl in labels:
            cls = lbl['cls']
            if cls not in class_map:
                continue
            
            val = class_map[cls]
            box = lbl['box'] # x, y, z, l, w, h, yaw
            
            # Get corners
            corners = get_corners_bev(box)
            
            # Project to pixels
            poly_pts = []
            for cp in corners:
                # Same transform as render_bev
                # u = cx - y / res
                # v = cy - x / res
                cu = int(cx - cp[1] / res)
                cv = int(cy - cp[0] / res)
                poly_pts.append((cu, cv))
            
            # Draw Polygon with class value
            draw.polygon(poly_pts, outline=val, fill=val) 
            
    return np.array(img)


def get_corners_3d(box):
    """
    Get 8 corners of the bounding box in 3D.
    box: [x, y, z, l, w, h, yaw]
    """
    x, y, z, l, w, h, yaw = box
    
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # Local corners (centered at 0)
    # l=x, w=y, h=z (assuming z is up)
    dx = l / 2
    dy = w / 2
    dz = h / 2

    # 8 corners: (+/-dx, +/-dy, +/-dz)
    corners_local = np.array([
        [dx, dy, dz], [dx, dy, -dz],
        [dx, -dy, dz], [dx, -dy, -dz],
        [-dx, dy, dz], [-dx, dy, -dz],
        [-dx, -dy, dz], [-dx, -dy, -dz]
    ])
    
    corners_global = (R @ corners_local.T).T + np.array([x, y, z + h/2])
    return corners_global


def render_range_view_mask(labels, width=1024, height=64, fov_up=45.0, fov_down=45.0, class_map=None):
    """
    Render Range View Label Mask (H, W).
    Projects 3D bounding boxes into the spherical range view.
    """
    if class_map is None:
        class_map = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}

    canvas = np.zeros((height, width), dtype=np.uint8)
    img = Image.fromarray(canvas, mode="L")
    draw = ImageDraw.Draw(img)

    fov_up_rad = fov_up * np.pi / 180.0
    fov_down_rad = fov_down * np.pi / 180.0
    fov = fov_up_rad + fov_down_rad

    if labels:
        for lbl in labels:
            cls = lbl['cls']
            if cls not in class_map:
                continue
            val = class_map[cls]
            box = lbl['box']

            corners = get_corners_3d(box)
            
            # Project corners
            pts_x = corners[:, 0]
            pts_y = corners[:, 1]
            pts_z = corners[:, 2] # z is up?

            depth = np.linalg.norm(corners[:, :3], axis=1)
            depth[depth == 0] = 0.001

            # Yaw: -pi to pi
            # Note: Matches render_range_view logic
            yaw = -np.arctan2(pts_y, pts_x)
            
            # Pitch
            pitch = np.arcsin(np.clip(pts_z / depth, -1.0, 1.0))

            # UV
            u = 0.5 * (yaw / np.pi + 1.0)
            v = 1.0 - (pitch + fov_down_rad) / fov
            
            u_img = (u * (width - 1)).astype(np.int32)
            v_img = (v * (height - 1)).astype(np.int32)
            
            # Wrap around handling (simple: if box spans edge, skip or clamp)
            # For simplicity, we just draw the convex hull of projected points
            
            # Form polygon
            poly_pts = list(zip(u_img, v_img))
            
            # Simple bounding box or convex hull is better, 
            # but ImageDraw.polygon draws the hull of ordered points? 
            # No, it draws a polygon connecting points.
            # We need the convex hull to fill the "blob".
            # Minimal approx: Bounding Box of the projected mask
            min_u, max_u = np.min(u_img), np.max(u_img)
            min_v, max_v = np.min(v_img), np.max(v_img)
            
            # Check for wraparound (e.g. min 0, max 1023)
            if (max_u - min_u) > width / 2:
                # Wrap-around case: The box crosses the -pi/pi boundary (left/right edge)
                # Split into two boxes: [min_u, width] and [0, max_u]
                
                # Part 1: Right side of image
                # Find all U points that are "large" (> width/2)
                mask_right = u_img > width / 2
                if np.any(mask_right):
                    u_r = u_img[mask_right]
                    v_r = v_img[mask_right]
                    if u_r.size > 0 and v_r.size > 0:
                        draw.rectangle([np.min(u_r), np.min(v_r), width, np.max(v_r)], fill=val, outline=val)
                    
                # Part 2: Left side of image
                # Find all U points that are "small" (< width/2)
                mask_left = u_img < width / 2
                if np.any(mask_left):
                    u_l = u_img[mask_left]
                    v_l = v_img[mask_left]
                    if u_l.size > 0 and v_l.size > 0:
                        draw.rectangle([0, np.min(v_l), np.max(u_l), np.max(v_l)], fill=val, outline=val)
                
                continue
            
            # Normal case
            # Actually fill the Convex Hull would be best. 
            # SciPy has ConvexHull, but we might not want the dependency just for this.
            # Let's draw the full box faces? 
            # Simpler: Draw the polygon valid for the 8 corners? No, 8 corners projection is scrambled.
            # Hack: Draw rectangle for now.
            draw.rectangle([min_u, min_v, max_u, max_v], fill=val, outline=val)

    return np.array(img)


def render_range_view(points, width=1024, height=64, fov_up=45.0, fov_down=45.0, stretch=6, max_dist=50.0, cmap_name="turbo"):
    """
    Convert (N, 3+) point cloud to a Range View (Spherical Projection).
    """
    if points.ndim != 2 or points.shape[1] < 3:
        return Image.new("L", (width, height))

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # 1. Calculate Depth and Angles
    depth = np.linalg.norm(points[:, :3], axis=1)
    depth[depth == 0] = 0.001
    
    # Yaw: -pi to pi (horizontal angle)
    yaw = -np.arctan2(y, x)
    
    # Pitch
    fov_up_rad = fov_up * np.pi / 180.0
    fov_down_rad = fov_down * np.pi / 180.0
    fov = fov_up_rad + fov_down_rad
    
    pitch = np.arcsin(np.clip(z / depth, -1.0, 1.0))

    # 2. Project to Pixel Coordinates
    u = 0.5 * (yaw / np.pi + 1.0)
    v = 1.0 - (pitch + fov_down_rad) / fov
    v = np.clip(v, 0.0, 1.0)
    
    u_idx = (u * (width - 1)).astype(np.int32)
    v_idx = (v * (height - 1)).astype(np.int32)
    
    # 3. Create Projection Canvas
    norm_depth = np.clip(np.clip(depth, 0, max_dist) / max_dist, 0, 1)
    
    canvas = np.zeros((height, width), dtype=np.uint8)
    intensity = ((1.0 - norm_depth) * 255).astype(np.uint8)
    canvas[v_idx, u_idx] = intensity

    try:
        cmap = cm.get_cmap(cmap_name)
        
        color_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        mask = canvas > 0
        valid_intensities = canvas[mask] / 255.0
        colors = cmap(valid_intensities)
        color_canvas[mask] = (colors[:, :3] * 255).astype(np.uint8)
        
        img = Image.fromarray(color_canvas, mode='RGB')
    except ImportError:
        img = Image.fromarray(canvas, mode='L')
    
    # 4. Vertical Stretch
    if stretch > 1:
         return img.resize((width, height * stretch), Image.Resampling.NEAREST)
    return img


def load_point_cloud_data(path):
    """Load point cloud from .bin (float32 kitti) or .npy files."""
    try:
        if path.endswith(".bin"):
            # Assume KITTI format: N x 4 (x, y, z, intensity) float32
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
            return points
        elif path.endswith(".npy"):
             points = np.load(path)
             return points
        elif path.endswith(".pcd"):
            # Placeholder for pcd parsing
            pass
    except Exception as e:
        logger.warning(f"Failed to load point cloud {path}: {e}")
    return None
