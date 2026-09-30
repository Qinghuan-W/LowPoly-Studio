import cv2
import math
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk


# ============================================================
# LowPoly Studio
#
# 100% Low Poly:
# - Original image is used only for analysis and polygon colours.
# - The original image is NEVER blended back into the output.
# - Any uncovered pixels use a flat global mean colour, not original pixels.
#
# Preview renders at reduced resolution for speed.
# Export renders at the ORIGINAL resolution.
# ============================================================


# -----------------------------
# File I/O
# -----------------------------

def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def save_png(path, image):
    success, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    if not success:
        raise RuntimeError("Failed to encode PNG.")
    encoded.tofile(str(path))


def normalize_map(data):
    data = data.astype(np.float32)
    minimum = float(np.min(data))
    maximum = float(np.max(data))

    if maximum - minimum < 1e-8:
        return np.zeros_like(data, dtype=np.float32)

    return (data - minimum) / (maximum - minimum)


# -----------------------------
# Feature analysis
# -----------------------------

def automatic_canny(gray):
    median = float(np.median(gray))
    sigma = 0.33

    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    if upper - lower < 30:
        lower = max(0, int(median - 30))
        upper = min(255, int(median + 30))

    return cv2.Canny(
        gray,
        lower,
        upper,
        L2gradient=True
    )


def grayscale_gradient(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(gx, gy)
    magnitude = cv2.GaussianBlur(magnitude, (5, 5), 0)

    return normalize_map(magnitude)


def colour_gradient(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    channels = cv2.split(lab)

    total = np.zeros(image.shape[:2], dtype=np.float32)

    for channel in channels:
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        total += cv2.magnitude(gx, gy)

    total = cv2.GaussianBlur(total, (5, 5), 0)
    return normalize_map(total)


def texture_map(gray):
    gray_float = gray.astype(np.float32)

    mean = cv2.GaussianBlur(gray_float, (0, 0), 3)
    mean_square = cv2.GaussianBlur(gray_float ** 2, (0, 0), 3)

    variance = np.maximum(mean_square - mean ** 2, 0)
    variance = cv2.GaussianBlur(variance, (7, 7), 0)

    return normalize_map(variance)


def create_importance_map(image, texture_weight):
    smooth = cv2.bilateralFilter(
        image,
        d=7,
        sigmaColor=28,
        sigmaSpace=28
    )

    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edge_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = automatic_canny(edge_gray)

    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    edge_map = dilated_edges.astype(np.float32) / 255.0

    gray_grad = grayscale_gradient(edge_gray)
    colour_grad = colour_gradient(smooth)
    texture = texture_map(edge_gray)

    # Structural features intentionally dominate texture.
    # texture_weight is exposed in the UI.
    gray_weight = 0.37
    colour_weight = 0.27
    edge_weight = 0.26

    importance = (
        gray_weight * gray_grad
        + colour_weight * colour_grad
        + float(texture_weight) * texture
        + edge_weight * edge_map
    )

    importance = cv2.GaussianBlur(importance, (5, 5), 0)
    importance = normalize_map(importance)

    return importance, edges, gray


def analyse_complexity(importance, edges):
    edge_density = float(np.mean(edges > 0))
    importance_mean = float(np.mean(importance))

    complexity = (
        0.65
        + edge_density * 3.2
        + importance_mean * 1.35
    )

    return float(np.clip(complexity, 0.75, 1.65))


# -----------------------------
# Point sampling
# -----------------------------

def add_border_points(points, width, height, spacing):
    points.extend([
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1)
    ])

    for x in range(0, width, spacing):
        points.append((x, 0))
        points.append((x, height - 1))

    for y in range(0, height, spacing):
        points.append((0, y))
        points.append((width - 1, y))


def add_corner_points(points, gray, maximum_corners, min_distance):
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max(1, int(maximum_corners)),
        qualityLevel=0.010,
        minDistance=max(1, int(min_distance)),
        blockSize=7,
        useHarrisDetector=False
    )

    if corners is None:
        return

    for corner in corners:
        x, y = corner.ravel()
        points.append((int(round(x)), int(round(y))))


def add_edge_points(points, edges, count, rng):
    ys, xs = np.where(edges > 0)

    total = len(xs)
    if total == 0:
        return

    count = min(int(count), total)

    if count <= 0:
        return

    indices = rng.choice(
        total,
        size=count,
        replace=False
    )

    for index in indices:
        points.append((int(xs[index]), int(ys[index])))


def add_architecture_line_points(
    points,
    edges,
    short_side,
    structure_strength
):
    if structure_strength <= 0:
        return

    min_line_length = max(30, int(short_side / 18))
    max_line_gap = max(8, int(short_side / 150))
    threshold = max(40, int(short_side / 30))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None:
        return

    # Compatible with both (N,1,4) and (N,4).
    lines = np.asarray(lines).reshape(-1, 4)

    line_data = []

    for x1, y1, x2, y2 in lines:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        length = math.hypot(x2 - x1, y2 - y1)
        line_data.append((length, x1, y1, x2, y2))

    line_data.sort(reverse=True, key=lambda item: item[0])

    max_lines = int(np.clip(65 * structure_strength, 0, 220))
    line_data = line_data[:max_lines]

    # 1.0 -> 5 points per selected line.
    points_per_segment = int(np.clip(round(4 * structure_strength), 2, 12))

    for _, x1, y1, x2, y2 in line_data:
        for i in range(points_per_segment + 1):
            t = i / points_per_segment
            x = int(round(x1 + (x2 - x1) * t))
            y = int(round(y1 + (y2 - y1) * t))
            points.append((x, y))


def add_importance_points(points, importance, count, rng):
    height, width = importance.shape

    probability = importance + 0.035
    probability = np.power(probability, 1.50)

    probability = probability.reshape(-1)
    probability /= np.sum(probability)

    total_pixels = width * height
    count = min(int(count), total_pixels)

    if count <= 0:
        return

    indices = rng.choice(
        total_pixels,
        size=count,
        replace=False,
        p=probability
    )

    ys = indices // width
    xs = indices % width

    for x, y in zip(xs, ys):
        points.append((int(x), int(y)))


def add_uniform_points(points, width, height, count, rng):
    for _ in range(max(0, int(count))):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        points.append((x, y))


def add_background_points(points, importance, count, rng, calmness):
    height, width = importance.shape
    count = max(0, int(count))

    if count <= 0:
        return

    calmness = float(np.clip(calmness, 0.0, 1.0))
    if calmness <= 1e-6:
        add_uniform_points(points, width, height, count, rng)
        return

    total_pixels = width * height
    cell_step = max(1.0, math.sqrt(total_pixels / count))
    xs = np.arange(cell_step / 2.0, width, cell_step)
    ys = np.arange(cell_step / 2.0, height, cell_step)

    candidates = []
    jitter = cell_step * (0.42 - 0.32 * calmness)

    for cy in ys:
        for cx in xs:
            best_point = None
            best_score = None

            trials = 4 if calmness < 0.5 else 6
            for _ in range(trials):
                px = cx + rng.uniform(-jitter, jitter)
                py = cy + rng.uniform(-jitter, jitter)
                x = int(np.clip(round(px), 0, width - 1))
                y = int(np.clip(round(py), 0, height - 1))

                local_importance = float(importance[y, x])
                center_bias = (abs(px - cx) + abs(py - cy)) / max(cell_step, 1.0)
                score = local_importance * (1.35 + 0.65 * calmness) + center_bias * (0.35 + 0.35 * calmness)

                if best_score is None or score < best_score:
                    best_score = score
                    best_point = (x, y)

            if best_point is not None:
                candidates.append(best_point)

    if len(candidates) > count:
        indices = rng.choice(len(candidates), size=count, replace=False)
        for index in indices:
            points.append(candidates[int(index)])
    else:
        points.extend(candidates)

    remaining = count - min(count, len(candidates))
    if remaining <= 0:
        return

    low_importance = np.power((1.0 - importance) + 0.03, 1.30 + 1.40 * calmness)
    probability = low_importance.reshape(-1)
    probability_sum = float(np.sum(probability))

    if probability_sum <= 1e-8:
        add_uniform_points(points, width, height, remaining, rng)
        return

    probability /= probability_sum

    indices = rng.choice(
        total_pixels,
        size=min(remaining, total_pixels),
        replace=False,
        p=probability
    )

    add_count = 0
    for index in indices:
        y = int(index // width)
        x = int(index % width)
        points.append((x, y))
        add_count += 1

    if add_count < remaining:
        add_uniform_points(points, width, height, remaining - add_count, rng)


def remove_near_duplicates(points, cell_size):
    cell_size = max(1, int(cell_size))

    cells = {}
    result = []

    for x, y in points:
        key = (x // cell_size, y // cell_size)

        if key in cells:
            continue

        cells[key] = True
        result.append((x, y))

    return result


# -----------------------------
# Delaunay and rendering
# -----------------------------

def create_subdiv(points, width, height):
    subdiv = cv2.Subdiv2D((0, 0, width, height))

    for x, y in points:
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))

        try:
            subdiv.insert((float(x), float(y)))
        except cv2.error:
            pass

    return subdiv


def delaunay_triangulation(points, width, height):
    subdiv = create_subdiv(points, width, height)
    return subdiv.getTriangleList()


def clean_polygon_vertices(polygon):
    cleaned = []

    for point in polygon:
        x = int(point[0])
        y = int(point[1])

        if not cleaned or cleaned[-1] != (x, y):
            cleaned.append((x, y))

    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    if len(cleaned) < 3:
        return None

    return np.array(cleaned, dtype=np.int32)


def voronoi_polygons_with_centers(points, width, height):
    subdiv = create_subdiv(points, width, height)
    facets, centers = subdiv.getVoronoiFacetList([])

    polygons = []
    valid_centers = []

    for facet, center in zip(facets, centers):
        if facet is None or len(facet) < 3:
            continue

        polygon = np.asarray(facet, dtype=np.float32).reshape(-1, 2)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        polygon = np.rint(polygon).astype(np.int32)
        polygon = clean_polygon_vertices(polygon)

        if polygon is None:
            continue

        polygons.append(polygon)
        valid_centers.append(np.asarray(center, dtype=np.float32))

    return polygons, valid_centers


def voronoi_polygons(points, width, height):
    polygons, _centers = voronoi_polygons_with_centers(points, width, height)
    return polygons


def polygon_centroid(polygon, width, height):
    moments = cv2.moments(polygon)

    if abs(moments["m00"]) > 1e-6:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx = float(np.mean(polygon[:, 0]))
        cy = float(np.mean(polygon[:, 1]))

    return (
        int(np.clip(round(cx), 0, width - 1)),
        int(np.clip(round(cy), 0, height - 1))
    )


def lloyd_relax_points(points, width, height, iterations):
    """Relax Voronoi sites toward their cell centroids.

    Border sites stay fixed so the tessellation keeps full-image coverage.
    """
    iterations = max(0, int(iterations))
    if iterations <= 0 or len(points) < 4:
        return list(points)

    current = [
        (
            int(np.clip(x, 0, width - 1)),
            int(np.clip(y, 0, height - 1))
        )
        for x, y in points
    ]

    for _ in range(iterations):
        polygons, centers = voronoi_polygons_with_centers(
            current,
            width,
            height
        )

        if not polygons:
            break

        # Subdiv2D returns each Voronoi center at its generating site.
        # Use rounded coordinates for fast site lookup; fall back to nearest.
        lookup = {point: index for index, point in enumerate(current)}
        point_array = np.asarray(current, dtype=np.float32)
        next_points = list(current)

        for polygon, center in zip(polygons, centers):
            center_key = (
                int(round(float(center[0]))),
                int(round(float(center[1])))
            )
            index = lookup.get(center_key)

            if index is None:
                distances = np.sum((point_array - center) ** 2, axis=1)
                index = int(np.argmin(distances))

            x, y = current[index]
            if x in (0, width - 1) or y in (0, height - 1):
                continue

            next_points[index] = polygon_centroid(
                polygon,
                width,
                height
            )

        # De-duplicate any sites that converged to the same pixel.
        seen = set()
        deduped = []
        for point in next_points:
            if point not in seen:
                seen.add(point)
                deduped.append(point)

        current = deduped

    return current


def create_colour_source(original, sigma):
    sigma = float(sigma)

    if sigma <= 0:
        return original.copy()

    return cv2.GaussianBlur(
        original,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT
    )


def polygon_average_colour(colour_source, polygon):
    x, y, w, h = cv2.boundingRect(polygon)

    if w <= 0 or h <= 0:
        return None

    roi = colour_source[y:y + h, x:x + w]

    local_polygon = polygon.copy()
    local_polygon[:, 0] -= x
    local_polygon[:, 1] -= y

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(
        mask,
        [local_polygon],
        255,
        lineType=cv2.LINE_8
    )

    mean = cv2.mean(roi, mask=mask)

    return (
        int(round(mean[0])),
        int(round(mean[1])),
        int(round(mean[2]))
    )


def triangle_average_colour(colour_source, triangle):
    return polygon_average_colour(colour_source, triangle)


def render_low_poly(original, triangles, colour_source, progress_callback=None):
    height, width = original.shape[:2]

    # IMPORTANT:
    # We do NOT use original.copy() here.
    # Any rare uncovered pixel is a flat mean colour,
    # so no original-image detail can leak into the result.
    mean = cv2.mean(colour_source)
    base_colour = np.array(
        [mean[0], mean[1], mean[2]],
        dtype=np.uint8
    )

    output = np.empty_like(original)
    output[:, :] = base_colour

    valid_count = 0
    total_items = max(1, len(triangles))

    for index, item in enumerate(triangles):
        if progress_callback is not None and (index % 50 == 0 or index == total_items - 1):
            progress_callback((index + 1) / total_items)
        triangle = np.array(
            item,
            dtype=np.float32
        ).reshape(3, 2)

        triangle = np.rint(triangle).astype(np.int32)

        xs = triangle[:, 0]
        ys = triangle[:, 1]

        if (
            np.any(xs < 0)
            or np.any(xs >= width)
            or np.any(ys < 0)
            or np.any(ys >= height)
        ):
            continue

        area = abs(cv2.contourArea(triangle))
        if area < 3:
            continue

        colour = triangle_average_colour(
            colour_source,
            triangle
        )

        if colour is None:
            continue

        cv2.fillConvexPoly(
            output,
            triangle,
            colour,
            lineType=cv2.LINE_8
        )

        valid_count += 1

    return output, valid_count



def render_voronoi_low_poly(original, polygons, colour_source, progress_callback=None):
    mean = cv2.mean(colour_source)
    base_colour = np.array(
        [mean[0], mean[1], mean[2]],
        dtype=np.uint8
    )

    output = np.empty_like(original)
    output[:, :] = base_colour

    valid_count = 0
    total_items = max(1, len(polygons))

    for index, polygon in enumerate(polygons):
        if progress_callback is not None and (index % 50 == 0 or index == total_items - 1):
            progress_callback((index + 1) / total_items)
        if polygon is None or len(polygon) < 3:
            continue

        area = abs(cv2.contourArea(polygon))
        if area < 6:
            continue

        colour = polygon_average_colour(
            colour_source,
            polygon
        )

        if colour is None:
            continue

        cv2.fillPoly(
            output,
            [polygon],
            colour,
            lineType=cv2.LINE_8
        )

        valid_count += 1

    return output, valid_count


def quantize_palette(image, palette_size, seed=0):
    palette_size = int(palette_size)

    if palette_size <= 1:
        return image

    flat = image.reshape(-1, 3).astype(np.float32)
    pixel_count = flat.shape[0]

    if pixel_count == 0:
        return image

    sample_limit = min(pixel_count, 20000)
    rng = np.random.default_rng(int(seed) + 17)

    if pixel_count > sample_limit:
        sample_indices = rng.choice(pixel_count, size=sample_limit, replace=False)
        sample = flat[sample_indices]
    else:
        sample = flat

    # OpenCV's k-means uses its own RNG for KMEANS_PP_CENTERS.
    # Seed it as well so the GUI seed reproduces the full result.
    cv2.setRNGSeed(int(seed))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 1.0)
    _compactness, _labels, centers = cv2.kmeans(
        sample,
        palette_size,
        None,
        criteria,
        2,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.clip(centers, 0, 255).astype(np.float32)
    recoloured = np.empty_like(flat)

    chunk_size = 50000
    for start in range(0, pixel_count, chunk_size):
        chunk = flat[start:start + chunk_size]
        distances = np.sum((chunk[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distances, axis=1)
        recoloured[start:start + len(chunk)] = centers[nearest]

    return np.clip(recoloured.reshape(image.shape), 0, 255).astype(np.uint8)


def soften_low_poly(image, strength, sigma):
    strength = float(np.clip(strength, 0.0, 1.0))
    sigma = float(sigma)

    if strength <= 0 or sigma <= 0:
        return image

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT
    )

    # Still 100% based on the Low Poly result.
    return cv2.addWeighted(
        image,
        1.0 - strength,
        blurred,
        strength,
        0
    )


def generate_low_poly(image, params, progress_callback=None):
    def report(progress):
        if progress_callback is not None:
            progress_callback(float(np.clip(progress, 0.0, 1.0)))

    report(0.02)

    height, width = image.shape[:2]
    short_side = min(width, height)
    long_side = max(width, height)
    aspect_factor = long_side / short_side

    rng = np.random.default_rng(int(params["seed"]))

    importance, edges, gray = create_importance_map(
        image,
        params["texture_weight"]
    )
    report(0.12)

    complexity = analyse_complexity(
        importance,
        edges
    )
    report(0.18)

    root_aspect = math.sqrt(aspect_factor)

    adaptive_count = int(
        760
        * complexity
        * root_aspect
        * params["polygon_density"]
    )

    edge_count = int(
        460
        * complexity
        * root_aspect
        * params["edge_detail"]
    )

    corner_count = int(
        240
        * complexity
        * root_aspect
        * params["corner_detail"]
    )

    background_count = int(
        75
        * root_aspect
        * params["background_density"]
    )

    adaptive_count = int(np.clip(adaptive_count, 120, 5200))
    edge_count = int(np.clip(edge_count, 60, 3200))
    corner_count = int(np.clip(corner_count, 40, 2000))
    background_count = int(np.clip(background_count, 8, 900))

    minimum_distance = max(
        4,
        int(short_side / (185 * max(params["corner_detail"], 0.50)))
    )

    border_spacing = max(
        24,
        int(short_side / 14)
    )

    points = []

    add_border_points(
        points,
        width,
        height,
        border_spacing
    )

    add_corner_points(
        points,
        gray,
        corner_count,
        minimum_distance
    )

    add_edge_points(
        points,
        edges,
        edge_count,
        rng
    )

    if params["architecture_lines"]:
        add_architecture_line_points(
            points,
            edges,
            short_side,
            params["structure_strength"]
        )

    add_importance_points(
        points,
        importance,
        adaptive_count,
        rng
    )

    add_background_points(
        points,
        importance,
        background_count,
        rng,
        params["background_calmness"]
    )
    report(0.34)

    quantization_size = max(
        2,
        int(short_side / (550 * max(params["polygon_density"], 0.55)))
    )

    points = remove_near_duplicates(
        points,
        quantization_size
    )
    report(0.42)

    geometry_mode = params.get("geometry_mode", "triangle")

    if geometry_mode == "voronoi":
        points = lloyd_relax_points(
            points,
            width,
            height,
            params.get("lloyd_iterations", 0)
        )
    report(0.50)

    if geometry_mode == "voronoi":
        geometries = voronoi_polygons(
            points,
            width,
            height
        )
    else:
        geometries = delaunay_triangulation(
            points,
            width,
            height
        )
    report(0.60)

    colour_source = create_colour_source(
        image,
        params["colour_softness"]
    )
    report(0.68)

    if geometry_mode == "voronoi":
        output, valid_count = render_voronoi_low_poly(
            image,
            geometries,
            colour_source,
            progress_callback=lambda value: report(0.68 + 0.22 * value)
        )
    else:
        output, valid_count = render_low_poly(
            image,
            geometries,
            colour_source,
            progress_callback=lambda value: report(0.68 + 0.22 * value)
        )

    report(0.91)

    if params["palette_size"] > 1:
        output = quantize_palette(
            output,
            params["palette_size"],
            seed=params["seed"]
        )

    report(0.96)

    output = soften_low_poly(
        output,
        params["final_softness"],
        params["final_sigma"]
    )

    report(0.99)

    info = {
        "complexity": complexity,
        "points": len(points),
        "geometry_count": len(geometries),
        "valid_polygons": valid_count,
        "geometry_mode": geometry_mode,
        "width": width,
        "height": height
    }

    report(1.0)
    return output, info


# -----------------------------
# Preview helpers
# -----------------------------

def resize_for_processing(image, max_side):
    height, width = image.shape[:2]
    longest = max(height, width)

    if longest <= max_side:
        return image.copy()

    scale = max_side / longest
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def cv_to_photoimage(image_bgr, max_width, max_height):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    width, height = pil.size
    scale = min(
        max_width / width,
        max_height / height,
        1.0
    )

    if scale < 1.0:
        pil = pil.resize(
            (
                max(1, int(width * scale)),
                max(1, int(height * scale))
            ),
            Image.Resampling.LANCZOS
        )

    return ImageTk.PhotoImage(pil)


# -----------------------------
# GUI
# -----------------------------

class LowPolyStudio:
    TRANSLATIONS = {
        "en": {
            "window_title": "LowPoly Studio",
            "subtitle": "100% polygon output · no original-image blending",
            "switch_language": "中文",
            "open_photo": "Open photo",
            "render_mode": "Geometry mode",
            "mode_triangle": "Triangle Low Poly",
            "mode_voronoi": "Voronoi Polygon",
            "parameters": "Parameters",
            "polygon_density": "Polygon density",
            "edge_detail": "Edge detail",
            "corner_detail": "Corner detail",
            "background_density": "Background density",
            "texture_influence": "Texture influence",
            "background_calmness": "Background calmness",
            "palette_colours": "Palette colours",
            "lloyd_relaxation": "Voronoi relaxation",
            "urban_line_enhance": "Urban straight-line enhancement",
            "structure_strength": "Structure strength",
            "polygon_colour_softness": "Polygon colour softness",
            "final_softness": "Final softness",
            "final_blur_radius": "Final blur radius",
            "seed": "Seed",
            "preview_max_side": "Preview max side",
            "render_preview": "Render preview",
            "export_full": "Export full-resolution PNG",
            "status_open": "Open a photo to begin.",
            "original": "Original",
            "lowpoly_preview": "Low Poly preview",
            "click_render": "Click “Render preview”",
            "dialog_open_title": "Open photo",
            "images": "Images",
            "all_files": "All files",
            "open_failed": "Open failed",
            "loaded": "Loaded {name} · {width} × {height}. Preview is reduced for speed; export keeps full resolution.",
            "no_photo_title": "No photo",
            "no_photo_message": "Open a photo first.",
            "rendering_preview": "Rendering preview...",
            "preview_ready": "Preview ready · {points} points · {polygons} polygons · complexity {complexity:.2f}",
            "export_title": "Export full-resolution Low Poly PNG",
            "png_image": "PNG image",
            "rendering_full": "Rendering FULL resolution. This may take a while...",
            "export_progress_title": "Export progress",
            "export_progress": "Exporting... {percent:.0f}%",
            "saved": "Saved {name} · {width} × {height} · {polygons} polygons",
            "export_complete": "Export complete",
            "export_message": "Full-resolution Low Poly PNG saved.\n\n{path}\n\nNo original-image detail was blended back into the result.",
            "render_failed": "Render failed",
        },
        "zh": {
            "window_title": "LowPoly Studio",
            "subtitle": "100% Low Poly 输出 · 不叠加任何原图细节",
            "switch_language": "English",
            "open_photo": "打开照片",
            "render_mode": "几何模式",
            "mode_triangle": "三角 Low Poly",
            "mode_voronoi": "Voronoi 多边形",
            "parameters": "参数",
            "polygon_density": "多边形密度",
            "edge_detail": "边缘细节",
            "corner_detail": "角点细节",
            "background_density": "背景密度",
            "texture_influence": "纹理影响",
            "background_calmness": "背景宁静度",
            "palette_colours": "色板颜色数",
            "lloyd_relaxation": "Voronoi 松弛",
            "urban_line_enhance": "城市直线结构增强",
            "structure_strength": "结构增强强度",
            "polygon_colour_softness": "多边形取色柔化",
            "final_softness": "最终柔化强度",
            "final_blur_radius": "最终模糊半径",
            "seed": "随机种子",
            "preview_max_side": "预览最长边",
            "render_preview": "生成预览",
            "export_full": "导出原分辨率 PNG",
            "status_open": "打开一张照片开始处理。",
            "original": "原图",
            "lowpoly_preview": "Low Poly 预览",
            "click_render": "点击“生成预览”",
            "dialog_open_title": "打开照片",
            "images": "图片",
            "all_files": "所有文件",
            "open_failed": "打开失败",
            "loaded": "已载入 {name} · {width} × {height}。预览会缩小以加快速度；导出仍保持原分辨率。",
            "no_photo_title": "还没有照片",
            "no_photo_message": "请先打开一张照片。",
            "rendering_preview": "正在生成预览...",
            "preview_ready": "预览完成 · {points} 个采样点 · {polygons} 个多边形 · 复杂度 {complexity:.2f}",
            "export_title": "导出原分辨率 Low Poly PNG",
            "png_image": "PNG 图片",
            "rendering_full": "正在以原分辨率渲染，可能需要一些时间...",
            "export_progress_title": "导出进度",
            "export_progress": "正在导出... {percent:.0f}%",
            "saved": "已保存 {name} · {width} × {height} · {polygons} 个多边形",
            "export_complete": "导出完成",
            "export_message": "原分辨率 Low Poly PNG 已保存。\n\n{path}\n\n最终结果没有混入或叠加任何原图细节。",
            "render_failed": "渲染失败",
        },
    }

    def __init__(self, root):
        self.root = root
        self.language = "zh"

        self.root.geometry("1500x900")
        self.root.minsize(1180, 720)

        self.original_image = None
        self.preview_result = None
        self.current_path = None

        self.original_photo = None
        self.result_photo = None

        self.last_preview_info = None
        self.busy = False
        self.export_progress_window = None
        self.export_progress_var = None
        self.export_progress_text_var = None

        self.vars = {
            "polygon_density": tk.DoubleVar(value=1.00),
            "edge_detail": tk.DoubleVar(value=1.00),
            "corner_detail": tk.DoubleVar(value=1.00),
            "background_density": tk.DoubleVar(value=1.00),
            "background_calmness": tk.DoubleVar(value=0.00),
            "texture_weight": tk.DoubleVar(value=0.14),
            "palette_size": tk.IntVar(value=0),
            "lloyd_iterations": tk.IntVar(value=0),
            "structure_strength": tk.DoubleVar(value=1.00),
            "colour_softness": tk.DoubleVar(value=1.00),
            "final_softness": tk.DoubleVar(value=0.22),
            "final_sigma": tk.DoubleVar(value=0.75),
            "architecture_lines": tk.BooleanVar(value=False),
            "seed": tk.IntVar(value=2026),
            "preview_size": tk.IntVar(value=1100),
            "geometry_mode": tk.StringVar(value="triangle"),
        }

        self.value_labels = {}

        self.build_ui()

    # ---------- i18n ----------

    def tr(self, key, **kwargs):
        text = self.TRANSLATIONS[self.language].get(key, key)
        if kwargs:
            return text.format(**kwargs)
        return text

    def mode_name(self, internal_name):
        mapping = {
            "triangle": "mode_triangle",
            "voronoi": "mode_voronoi",
        }
        return self.tr(mapping.get(internal_name, internal_name))

    def toggle_language(self):
        if self.busy:
            return

        self.language = "en" if self.language == "zh" else "zh"
        self.rebuild_ui()

    def rebuild_ui(self):
        for child in self.root.winfo_children():
            child.destroy()

        self.value_labels = {}
        self.build_ui()

        if self.original_image is not None:
            self.root.after(20, self.display_original)

            if self.preview_result is not None:
                self.root.after(
                    30,
                    lambda: self.display_result(self.preview_result)
                )
            else:
                self.result_label.configure(
                    image="",
                    text=self.tr("click_render")
                )

        self.refresh_status_for_language()

    def refresh_status_for_language(self):
        if self.original_image is None or self.current_path is None:
            self.status_var.set(self.tr("status_open"))
            return

        if self.preview_result is not None and self.last_preview_info is not None:
            info = self.last_preview_info
            self.status_var.set(
                self.tr(
                    "preview_ready",
                    points=info["points"],
                    polygons=info["valid_polygons"],
                    complexity=info["complexity"]
                )
            )
            return

        height, width = self.original_image.shape[:2]
        self.status_var.set(
            self.tr(
                "loaded",
                name=self.current_path.name,
                width=width,
                height=height
            )
        )

    # ---------- UI construction ----------

    def build_ui(self):
        self.root.title(self.tr("window_title"))

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Scrollable left control panel.
        left_shell = ttk.Frame(self.root)
        left_shell.grid(
            row=0,
            column=0,
            sticky="nsew"
        )
        left_shell.rowconfigure(0, weight=1)
        left_shell.columnconfigure(0, weight=1)

        left_canvas = tk.Canvas(
            left_shell,
            width=370,
            highlightthickness=0,
            borderwidth=0
        )
        left_scrollbar = ttk.Scrollbar(
            left_shell,
            orient="vertical",
            command=left_canvas.yview
        )
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.grid(
            row=0,
            column=0,
            sticky="nsew"
        )
        left_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        left = ttk.Frame(
            left_canvas,
            padding=12
        )
        left_window = left_canvas.create_window(
            (0, 0),
            window=left,
            anchor="nw"
        )

        def update_left_scrollregion(_event=None):
            bbox = left_canvas.bbox("all")
            if bbox is not None:
                left_canvas.configure(scrollregion=bbox)

        def resize_left_content(event):
            left_canvas.itemconfigure(
                left_window,
                width=event.width
            )

        def scroll_left(event):
            if getattr(event, "num", None) == 4:
                left_canvas.yview_scroll(-1, "units")
                return "break"
            if getattr(event, "num", None) == 5:
                left_canvas.yview_scroll(1, "units")
                return "break"

            delta = getattr(event, "delta", 0)
            if delta:
                step = -int(delta / 120)
                if step == 0:
                    step = -1 if delta > 0 else 1
                left_canvas.yview_scroll(step, "units")
                return "break"

        def bind_left_mousewheel(_event=None):
            left_canvas.bind_all("<MouseWheel>", scroll_left)
            left_canvas.bind_all("<Button-4>", scroll_left)
            left_canvas.bind_all("<Button-5>", scroll_left)

        def unbind_left_mousewheel(_event=None):
            left_canvas.unbind_all("<MouseWheel>")
            left_canvas.unbind_all("<Button-4>")
            left_canvas.unbind_all("<Button-5>")

        left.bind("<Configure>", update_left_scrollregion)
        left_canvas.bind("<Configure>", resize_left_content)
        left_shell.bind("<Enter>", bind_left_mousewheel)
        left_shell.bind("<Leave>", unbind_left_mousewheel)

        # Keep references alive and available after UI construction.
        self.left_canvas = left_canvas
        self.left_scrollbar = left_scrollbar

        right = ttk.Frame(
            self.root,
            padding=(0, 12, 12, 12)
        )
        right.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=1)
        right.rowconfigure(0, weight=1)

        # Header
        header = ttk.Frame(left)
        header.pack(fill="x")

        title = ttk.Label(
            header,
            text="LowPoly Studio",
            font=("Segoe UI", 18, "bold")
        )
        title.pack(
            side="left",
            anchor="w"
        )

        self.language_button = ttk.Button(
            header,
            text=self.tr("switch_language"),
            command=self.toggle_language,
            width=9
        )
        self.language_button.pack(
            side="right",
            padx=(8, 0)
        )

        subtitle = ttk.Label(
            left,
            text=self.tr("subtitle"),
            foreground="#666666"
        )
        subtitle.pack(
            anchor="w",
            pady=(2, 8)
        )

        mode_box = ttk.LabelFrame(
            left,
            text=self.tr("render_mode"),
            padding=6
        )
        mode_box.pack(
            fill="x",
            pady=(0, 10)
        )

        mode_row = ttk.Frame(mode_box)
        mode_row.pack(fill="x")

        self.mode_triangle_button = tk.Radiobutton(
            mode_row,
            text=self.mode_name("triangle"),
            variable=self.vars["geometry_mode"],
            value="triangle",
            indicatoron=False,
            command=self.on_mode_changed,
            padx=10,
            pady=6
        )
        self.mode_triangle_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=(0, 3)
        )

        self.mode_voronoi_button = tk.Radiobutton(
            mode_row,
            text=self.mode_name("voronoi"),
            variable=self.vars["geometry_mode"],
            value="voronoi",
            indicatoron=False,
            command=self.on_mode_changed,
            padx=10,
            pady=6
        )
        self.mode_voronoi_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=(3, 0)
        )

        ttk.Button(
            left,
            text=self.tr("open_photo"),
            command=self.open_image
        ).pack(
            fill="x",
            pady=(0, 8)
        )

        # Parameters
        param_box = ttk.LabelFrame(
            left,
            text=self.tr("parameters"),
            padding=8
        )
        param_box.pack(fill="x")

        self.add_slider(
            param_box,
            self.tr("polygon_density"),
            "polygon_density",
            0.20,
            3.00,
            0.01
        )

        self.add_slider(
            param_box,
            self.tr("edge_detail"),
            "edge_detail",
            0.20,
            3.00,
            0.01
        )

        self.add_slider(
            param_box,
            self.tr("corner_detail"),
            "corner_detail",
            0.20,
            3.00,
            0.01
        )

        self.add_slider(
            param_box,
            self.tr("background_density"),
            "background_density",
            0.10,
            3.00,
            0.01
        )

        self.add_slider(
            param_box,
            self.tr("background_calmness"),
            "background_calmness",
            0.00,
            1.00,
            0.01
        )

        self.add_slider(
            param_box,
            self.tr("texture_influence"),
            "texture_weight",
            0.00,
            0.80,
            0.01
        )

        separator = ttk.Separator(
            param_box,
            orient="horizontal"
        )
        separator.pack(
            fill="x",
            pady=7
        )

        ttk.Checkbutton(
            param_box,
            text=self.tr("urban_line_enhance"),
            variable=self.vars["architecture_lines"]
        ).pack(
            anchor="w",
            pady=(0, 4)
        )

        self.add_slider(
            param_box,
            self.tr("structure_strength"),
            "structure_strength",
            0.00,
            3.00,
            0.01
        )

        separator = ttk.Separator(
            param_box,
            orient="horizontal"
        )
        separator.pack(
            fill="x",
            pady=7
        )

        self.add_slider(
            param_box,
            self.tr("polygon_colour_softness"),
            "colour_softness",
            0.00,
            4.00,
            0.01
        )

        self.add_slider(
            param_box,
            self.tr("final_softness"),
            "final_softness",
            0.00,
            1.00,
            0.01
        )

        self.add_slider(
            param_box,
            self.tr("final_blur_radius"),
            "final_sigma",
            0.00,
            3.00,
            0.01
        )

        self.add_slider(
            param_box,
            self.tr("palette_colours"),
            "palette_size",
            0,
            24,
            1
        )

        self.add_slider(
            param_box,
            self.tr("lloyd_relaxation"),
            "lloyd_iterations",
            0,
            5,
            1
        )

        # Seed and preview size
        misc = ttk.Frame(param_box)
        misc.pack(
            fill="x",
            pady=(8, 0)
        )

        ttk.Label(
            misc,
            text=self.tr("seed")
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        seed_entry = ttk.Spinbox(
            misc,
            from_=0,
            to=999999,
            textvariable=self.vars["seed"],
            width=10
        )
        seed_entry.grid(
            row=0,
            column=1,
            sticky="e",
            padx=(10, 0)
        )

        ttk.Label(
            misc,
            text=self.tr("preview_max_side")
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(6, 0)
        )

        preview_combo = ttk.Combobox(
            misc,
            textvariable=self.vars["preview_size"],
            values=(500, 700, 900, 1100, 1400, 1800, 2200),
            state="readonly",
            width=8
        )
        preview_combo.grid(
            row=1,
            column=1,
            sticky="e",
            padx=(10, 0),
            pady=(6, 0)
        )

        misc.columnconfigure(0, weight=1)

        # Actions
        action_box = ttk.Frame(left)
        action_box.pack(
            fill="x",
            pady=(10, 0)
        )

        self.preview_button = ttk.Button(
            action_box,
            text=self.tr("render_preview"),
            command=self.render_preview
        )
        self.preview_button.pack(
            fill="x",
            pady=(0, 6)
        )

        self.export_button = ttk.Button(
            action_box,
            text=self.tr("export_full"),
            command=self.export_full
        )
        self.export_button.pack(fill="x")

        self.progress = ttk.Progressbar(
            left,
            mode="indeterminate"
        )
        self.progress.pack(
            fill="x",
            pady=(12, 4)
        )

        self.status_var = tk.StringVar(
            value=self.tr("status_open")
        )

        ttk.Label(
            left,
            textvariable=self.status_var,
            wraplength=345,
            foreground="#555555"
        ).pack(fill="x")

        # Preview panes
        original_frame = ttk.LabelFrame(
            right,
            text=self.tr("original"),
            padding=8
        )
        original_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 6)
        )

        result_frame = ttk.LabelFrame(
            right,
            text=self.tr("lowpoly_preview"),
            padding=8
        )
        result_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(6, 0)
        )

        original_frame.rowconfigure(0, weight=1)
        original_frame.columnconfigure(0, weight=1)

        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)

        self.original_label = ttk.Label(
            original_frame,
            anchor="center"
        )
        self.original_label.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        self.result_label = ttk.Label(
            result_frame,
            anchor="center"
        )
        self.result_label.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        if self.original_image is not None and self.preview_result is None:
            self.result_label.configure(
                text=self.tr("click_render")
            )

    def on_mode_changed(self):
        # Geometry mode is already updated automatically by the Radiobutton variable.
        # No preset state is used in the manual-only version.
        pass

    def add_slider(
        self,
        parent,
        label_text,
        key,
        minimum,
        maximum,
        resolution
    ):
        outer = ttk.Frame(parent)
        outer.pack(
            fill="x",
            pady=3
        )

        top = ttk.Frame(outer)
        top.pack(fill="x")

        ttk.Label(
            top,
            text=label_text
        ).pack(side="left")

        value_label = ttk.Label(
            top,
            width=7,
            anchor="e"
        )
        value_label.pack(side="right")

        self.value_labels[key] = value_label

        scale = ttk.Scale(
            outer,
            from_=minimum,
            to=maximum,
            variable=self.vars[key],
            orient="horizontal",
            command=lambda value, k=key, r=resolution: self.on_scale(k, value, r)
        )
        scale.pack(fill="x")

        self.update_value_label(key)

    def on_scale(self, key, value, resolution):
        raw = float(value)

        if resolution >= 1:
            rounded = round(raw)
        else:
            decimals = max(
                0,
                int(round(-math.log10(resolution)))
            )
            rounded = round(raw, decimals)

        self.vars[key].set(rounded)
        self.update_value_label(key)

    def update_value_label(self, key):
        value = self.vars[key].get()

        if key == "texture_weight":
            text = f"{float(value):.2f}"
        elif key in {
            "colour_softness",
            "final_softness",
            "final_sigma",
            "polygon_density",
            "edge_detail",
            "corner_detail",
            "background_density",
            "background_calmness",
            "structure_strength"
        }:
            text = f"{float(value):.2f}"
        elif key in {"palette_size", "lloyd_iterations"}:
            text = str(int(value))
        else:
            text = str(value)

        if key in self.value_labels:
            self.value_labels[key].configure(text=text)

    # ---------- state ----------

    def get_params(self):
        return {
            "polygon_density": float(self.vars["polygon_density"].get()),
            "edge_detail": float(self.vars["edge_detail"].get()),
            "corner_detail": float(self.vars["corner_detail"].get()),
            "background_density": float(self.vars["background_density"].get()),
            "background_calmness": float(self.vars["background_calmness"].get()),
            "texture_weight": float(self.vars["texture_weight"].get()),
            "palette_size": int(self.vars["palette_size"].get()),
            "lloyd_iterations": int(self.vars["lloyd_iterations"].get()),
            "structure_strength": float(self.vars["structure_strength"].get()),
            "colour_softness": float(self.vars["colour_softness"].get()),
            "final_softness": float(self.vars["final_softness"].get()),
            "final_sigma": float(self.vars["final_sigma"].get()),
            "architecture_lines": bool(self.vars["architecture_lines"].get()),
            "seed": int(self.vars["seed"].get()),
            "geometry_mode": self.vars["geometry_mode"].get(),
        }

    def set_busy(self, busy, message=None, progress_mode="indeterminate"):
        self.busy = busy

        if busy:
            self.preview_button.configure(state="disabled")
            self.export_button.configure(state="disabled")
            self.language_button.configure(state="disabled")
            if progress_mode == "indeterminate":
                self.progress.configure(mode="indeterminate")
                self.progress.start(12)
            else:
                self.progress.stop()
                self.progress.configure(mode="determinate")
                try:
                    self.progress.configure(value=0)
                except tk.TclError:
                    pass
        else:
            self.preview_button.configure(state="normal")
            self.export_button.configure(state="normal")
            self.language_button.configure(state="normal")
            self.progress.stop()
            self.progress.configure(mode="indeterminate")

        if message is not None:
            self.status_var.set(message)

    def show_export_progress_dialog(self):
        self.hide_export_progress_dialog()

        self.export_progress_window = tk.Toplevel(self.root)
        self.export_progress_window.title(self.tr("export_progress_title"))
        self.export_progress_window.transient(self.root)
        self.export_progress_window.resizable(False, False)
        self.export_progress_window.protocol(
            "WM_DELETE_WINDOW",
            lambda: None
        )

        body = ttk.Frame(self.export_progress_window, padding=14)
        body.pack(fill="both", expand=True)

        ttk.Label(
            body,
            text=self.tr("rendering_full"),
            wraplength=340,
            justify="left"
        ).pack(anchor="w")

        self.export_progress_var = tk.DoubleVar(value=0.0)
        self.export_progress_text_var = tk.StringVar(
            value=self.tr("export_progress", percent=0)
        )

        ttk.Progressbar(
            body,
            mode="determinate",
            maximum=100,
            variable=self.export_progress_var,
            length=340
        ).pack(fill="x", pady=(12, 8))

        ttk.Label(
            body,
            textvariable=self.export_progress_text_var,
            anchor="center"
        ).pack(fill="x")

        self.export_progress_window.update_idletasks()
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()
        win_w = self.export_progress_window.winfo_width()
        win_h = self.export_progress_window.winfo_height()
        x = root_x + max(0, (root_w - win_w) // 2)
        y = root_y + max(0, (root_h - win_h) // 2)
        self.export_progress_window.geometry(f"+{x}+{y}")
        self.export_progress_window.grab_set()

    def hide_export_progress_dialog(self):
        if self.export_progress_window is not None:
            try:
                self.export_progress_window.grab_release()
            except tk.TclError:
                pass
            try:
                self.export_progress_window.destroy()
            except tk.TclError:
                pass
        self.export_progress_window = None
        self.export_progress_var = None
        self.export_progress_text_var = None

    def update_export_progress(self, progress):
        percent = float(np.clip(progress, 0.0, 1.0)) * 100.0
        self.status_var.set(
            self.tr("export_progress", percent=percent)
        )

        try:
            self.progress.configure(mode="determinate", value=percent)
        except tk.TclError:
            pass

        if self.export_progress_var is not None:
            self.export_progress_var.set(percent)
        if self.export_progress_text_var is not None:
            self.export_progress_text_var.set(
                self.tr("export_progress", percent=percent)
            )

    # ---------- image display ----------

    def display_original(self):
        if self.original_image is None:
            return

        self.root.update_idletasks()

        width = max(
            400,
            self.original_label.winfo_width() - 12
        )
        height = max(
            400,
            self.original_label.winfo_height() - 12
        )

        self.original_photo = cv_to_photoimage(
            self.original_image,
            width,
            height
        )

        self.original_label.configure(
            image=self.original_photo,
            text=""
        )

    def display_result(self, image):
        self.root.update_idletasks()

        width = max(
            400,
            self.result_label.winfo_width() - 12
        )
        height = max(
            400,
            self.result_label.winfo_height() - 12
        )

        self.result_photo = cv_to_photoimage(
            image,
            width,
            height
        )

        self.result_label.configure(
            image=self.result_photo,
            text=""
        )

    # ---------- actions ----------

    def open_image(self):
        if self.busy:
            return

        path = filedialog.askopenfilename(
            title=self.tr("dialog_open_title"),
            filetypes=[
                (
                    self.tr("images"),
                    "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.JPG *.JPEG *.PNG *.TIF *.TIFF *.WEBP"
                ),
                (self.tr("all_files"), "*.*")
            ]
        )

        if not path:
            return

        try:
            image = read_image(path)
        except Exception as exc:
            messagebox.showerror(
                self.tr("open_failed"),
                str(exc)
            )
            return

        self.current_path = Path(path)
        self.original_image = image
        self.preview_result = None
        self.last_preview_info = None

        self.result_label.configure(
            image="",
            text=self.tr("click_render")
        )

        self.display_original()

        height, width = image.shape[:2]

        self.status_var.set(
            self.tr(
                "loaded",
                name=self.current_path.name,
                width=width,
                height=height
            )
        )

    def render_preview(self):
        if self.busy:
            return

        if self.original_image is None:
            messagebox.showinfo(
                self.tr("no_photo_title"),
                self.tr("no_photo_message")
            )
            return

        params = self.get_params()
        preview_size = int(self.vars["preview_size"].get())

        processing_image = resize_for_processing(
            self.original_image,
            preview_size
        )

        self.set_busy(
            True,
            self.tr("rendering_preview"),
            progress_mode="indeterminate"
        )

        thread = threading.Thread(
            target=self._preview_worker,
            args=(processing_image, params),
            daemon=True
        )
        thread.start()

    def _preview_worker(self, image, params):
        try:
            result, info = generate_low_poly(
                image,
                params
            )

            self.root.after(
                0,
                lambda: self._preview_done(
                    result,
                    info
                )
            )

        except Exception:
            error = traceback.format_exc()

            self.root.after(
                0,
                lambda: self._worker_error(error)
            )

    def _preview_done(self, result, info):
        self.preview_result = result
        self.last_preview_info = info
        self.display_result(result)

        self.set_busy(
            False,
            self.tr(
                "preview_ready",
                points=info["points"],
                polygons=info["valid_polygons"],
                complexity=info["complexity"]
            )
        )

    def export_full(self):
        if self.busy:
            return

        if self.original_image is None or self.current_path is None:
            messagebox.showinfo(
                self.tr("no_photo_title"),
                self.tr("no_photo_message")
            )
            return

        default_name = (
            self.current_path.stem
            + "_lowpoly.png"
        )

        path = filedialog.asksaveasfilename(
            title=self.tr("export_title"),
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                (self.tr("png_image"), "*.png")
            ]
        )

        if not path:
            return

        params = self.get_params()
        image = self.original_image.copy()

        self.set_busy(
            True,
            self.tr("export_progress", percent=0),
            progress_mode="determinate"
        )
        self.show_export_progress_dialog()
        self.update_export_progress(0.0)

        thread = threading.Thread(
            target=self._export_worker,
            args=(
                image,
                params,
                Path(path)
            ),
            daemon=True
        )
        thread.start()

    def _export_worker(
        self,
        image,
        params,
        output_path
    ):
        try:
            def progress_callback(progress):
                # Rendering uses 0-98% of the export progress.
                # Reserve the last 2% for PNG encoding/writing and completion.
                export_progress = float(np.clip(progress, 0.0, 1.0)) * 0.98
                self.root.after(
                    0,
                    lambda p=export_progress: self.update_export_progress(p)
                )

            result, info = generate_low_poly(
                image,
                params,
                progress_callback=progress_callback
            )

            self.root.after(
                0,
                lambda: self.update_export_progress(0.99)
            )

            save_png(
                output_path,
                result
            )

            self.root.after(
                0,
                lambda: self._export_done(
                    output_path,
                    info
                )
            )

        except Exception:
            error = traceback.format_exc()

            self.root.after(
                0,
                lambda: self._worker_error(error)
            )

    def _export_done(self, output_path, info):
        self.update_export_progress(1.0)
        self.hide_export_progress_dialog()
        self.set_busy(
            False,
            self.tr(
                "saved",
                name=output_path.name,
                width=info["width"],
                height=info["height"],
                polygons=info["valid_polygons"]
            )
        )

        messagebox.showinfo(
            self.tr("export_complete"),
            self.tr(
                "export_message",
                path=output_path
            )
        )

    def _worker_error(self, error):
        self.hide_export_progress_dialog()
        self.set_busy(
            False,
            self.tr("render_failed")
        )

        messagebox.showerror(
            self.tr("render_failed"),
            error
        )

def main():
    root = tk.Tk()

    # Use native ttk styling where possible.
    try:
        style = ttk.Style(root)

        if "vista" in style.theme_names():
            style.theme_use("vista")

    except Exception:
        pass

    app = LowPolyStudio(root)

    # Refresh original preview after window layout settles.
    root.after(
        300,
        app.display_original
    )

    root.mainloop()


if __name__ == "__main__":
    main()
