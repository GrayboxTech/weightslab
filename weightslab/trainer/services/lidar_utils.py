import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
logger = logging.getLogger(__name__)


def render_lidar(points, config):
    """
    Main entry point for LiDAR visualization. Dispatch options driven by config.
    """
    mode = config.get("mode", "bev")
    cmap_name = config.get("cmap", "turbo")
    max_range = config.get("max_range", 50.0)

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
            cmap_name=cmap_name
        )


def render_bev(points, res=0.1, size=800, cx=400, cy=400, max_dist=50.0, cmap_name="turbo"):
    """
    Render Top-Down Bird's Eye View.
    X -> Right (Image X)
    Y -> Forward (Image Y, inverted for pixel coords)
    Z -> Height (Color)
    """
    if points.size == 0:
        return Image.new("RGB", (size, size))
        
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2] # Use Z for color

    # 1. Map to pixels
    # Image Coords: (0,0) top-left.
    # World: +Y Forward.
    # Map: World(0,0) -> Image(cx, cy)
    # u = cx + x / res
    # v = cy - y / res (Invert Y because image Y goes down)
    
    u = (cx + x / res).astype(np.int32)
    v = (cy - y / res).astype(np.int32)
    
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
             rgb = (colors[:, :3] * 255).astype(np.uint8)
             
             # Draw points. Simple painter's algorithm (later points overwrite)
             # For better results in dense clouds, could sort filters or use max-z buffer.
             canvas[v, u] = rgb
             
        except ImportError:
             # Fallback white
             canvas[v, u] = 255
    
    return Image.fromarray(canvas, mode="RGB")


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
