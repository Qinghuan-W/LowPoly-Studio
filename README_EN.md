# LowPoly Studio

[中文](README.md) | [English](README_EN.md)

LowPoly Studio is a desktop image stylisation tool that converts photographs into **100% polygon-based Low Poly artwork**.

The application supports two geometry modes:

- **Triangle Low Poly** — content-aware Delaunay triangulation
- **Voronoi Polygon** — Voronoi tessellation with optional Lloyd relaxation

Unlike approaches that blend the original photograph back into the final image, LowPoly Studio uses the source image only for **analysis and polygon colour estimation**. The final output is rendered entirely from polygons.

<img width="2559" height="1378" alt="image" src="https://github.com/user-attachments/assets/230be6d7-6f73-4181-9efa-23405178f1a0" />


<img width="2559" height="1380" alt="image" src="https://github.com/user-attachments/assets/daf42cb7-3f60-44f1-8ceb-1d8171fc141a" />



---

## Features

- **100% polygon output**
  - The original image is never blended back into the final result.
  - Uncovered pixels use a flat global mean colour instead of original-image pixels.

- **Two geometry modes**
  - Delaunay triangle mesh
  - Voronoi polygon tessellation

- **Content-aware point sampling**
  - grayscale gradients
  - colour gradients
  - Canny edges
  - local texture
  - Shi–Tomasi corner detection
  - optional Hough line sampling for architectural scenes

- **Voronoi Lloyd relaxation**
  - Regularises Voronoi cell distribution.
  - Border sites remain fixed to preserve image coverage.

- **Adjustable artistic controls**
  - polygon density
  - edge detail
  - corner detail
  - background density
  - background calmness
  - texture influence
  - structure enhancement
  - polygon colour softness
  - final softness
  - palette size
  - Voronoi relaxation iterations
  - random seed

- **Deterministic generation**
  - NumPy and OpenCV random generators are seeded so that the same image, parameters, and seed reproduce the same result.

- **Preview and full-resolution export**
  - Preview can be rendered at a reduced resolution for speed.
  - PNG export is generated at the original image resolution.

- **Export progress dialog**
  - Visual progress from 0% to 100%.

- **Bilingual interface**
  - Chinese
  - English

---

## Requirements

- Python 3.10+ recommended
- OpenCV
- NumPy
- Pillow
- Tkinter

Install the Python dependencies with:

```bash
pip install opencv-python numpy Pillow
```

Tkinter is normally bundled with standard Python distributions.

On Linux, it may need to be installed separately, for example:

```bash
sudo apt install python3-tk
```

---

## Running the Application

Clone the repository:

```bash
git clone https://github.com/Qinghuan-W/LowPoly-Studio.git
cd LowPoly-Studio
```

Run:

```bash
python lowpoly_studio.py
```

On systems where Python 3 is invoked explicitly:

```bash
python3 lowpoly_studio.py
```

---

# Algorithm Overview

The rendering pipeline is:

```text
Input image
    │
    ▼
Bilateral smoothing
    │
    ▼
Feature analysis
 ├─ grayscale gradient
 ├─ colour gradient
 ├─ Canny edge map
 └─ local texture map
    │
    ▼
Importance map
    │
    ▼
Content-aware point sampling
 ├─ border points
 ├─ Shi–Tomasi corners
 ├─ edge points
 ├─ optional Hough line points
 ├─ importance-weighted points
 └─ calm-background points
    │
    ▼
Point de-duplication
    │
    ├───────────────┐
    ▼               ▼
Delaunay         Voronoi
triangulation    tessellation
                    │
                    ▼
              Lloyd relaxation
              (optional)
    │               │
    └───────┬───────┘
            ▼
    Mean colour per polygon
            │
            ▼
    Optional palette quantisation
            │
            ▼
    Optional Low-Poly-only softening
            │
            ▼
       Final PNG output
```

---

# Mathematical Model

## 1. Feature Maps

Let the input image be $I(x,y)$.

Before feature extraction, the image is smoothed using a bilateral filter so that weak image noise is reduced while strong boundaries are preserved.

### Grayscale gradient

For grayscale image $Y$, horizontal and vertical Sobel responses are:

```math
G_x = S_x * Y
```
```math
G_y = S_y * Y
```
The gradient magnitude is:

```math
G_{\text{gray}}(x,y)
=
\sqrt{G_x(x,y)^2 + G_y(x,y)^2}
```
The result is Gaussian-smoothed and normalised into $[0,1]$.

---

## 2. Colour Gradient

The image is converted from BGR to CIELAB colour space.

For each LAB channel $k$:

```math
G_k(x,y)
=
\sqrt{
G_{x,k}(x,y)^2
+
G_{y,k}(x,y)^2
}
```
The colour-gradient response is:

```math
G_{\text{colour}}(x,y)
=
\mathrm{Norm}
\left(
\sum_k G_k(x,y)
\right)
```
LAB is used because it separates luminance from chromatic components more effectively than operating directly on BGR channels.

---

## 3. Local Texture

Texture is estimated from local variance.

For grayscale intensity $Y$:

```math
\mu(x,y) = \mathcal{G}_{\sigma}(Y)
```
```math
\mu_2(x,y) = \mathcal{G}_{\sigma}(Y^2)
```
```math
V(x,y)
=
\max
\left(
\mu_2(x,y)-\mu(x,y)^2,
0
\right)
```
The texture map is then smoothed and normalised:

```math
T(x,y)=\mathrm{Norm}(V(x,y))
```
---

## 4. Importance Map

LowPoly Studio combines grayscale structure, colour structure, texture, and Canny edges into a single importance field.

The implementation uses:

```math
I_{\text{raw}}
=
0.37G_{\text{gray}}
+
0.27G_{\text{colour}}
+
w_tT
+
0.26E
```
where:

- $G_{\text{gray}}$ is the normalised grayscale gradient
- $G_{\text{colour}}$ is the normalised LAB colour gradient
- $T$ is the texture map
- $w_t$ is the user-controlled texture influence
- $E$ is the dilated Canny edge map

The final importance map is:

```math
I
=
\mathrm{Norm}
\left(
\mathcal{G}(I_{\text{raw}})
\right)
```
The map therefore assigns more sampling density to visually significant regions.

---

## 5. Image Complexity

The application estimates image complexity using edge density and average importance:

```math
C_{\text{raw}}
=
0.65
+
3.2d_e
+
1.35\bar{I}
```
where:

```math
d_e
=
\frac{\text{number of edge pixels}}
{\text{number of image pixels}}
```
and:

```math
\bar{I}
=
\frac{1}{WH}
\sum_{x,y}I(x,y)
```
The value is clamped:

```math
C
=
\mathrm{clip}
\left(
C_{\text{raw}},
0.75,
1.65
\right)
```
This automatically allocates more geometry to visually complex images.

---

## 6. Adaptive Point Counts

Let:

```math
A
=
\frac{\max(W,H)}
{\min(W,H)}
```
and:

```math
R=\sqrt{A}
```
The approximate number of sampled points in each category is:

### Importance points

```math
N_{\text{adaptive}}
=
760
\cdot
C
\cdot
R
\cdot
D_p
```
### Edge points

```math
N_{\text{edge}}
=
460
\cdot
C
\cdot
R
\cdot
D_e
```
### Corner points

```math
N_{\text{corner}}
=
240
\cdot
C
\cdot
R
\cdot
D_c
```
### Background points

```math
N_{\text{background}}
=
75
\cdot
R
\cdot
D_b
```
where $D_p,D_e,D_c,D_b$ are the user-controlled density parameters.

The implementation clamps these values to practical ranges to control memory usage and rendering cost.

---

## 7. Importance-Weighted Sampling

Pixels are sampled according to the importance map.

The sampling probability is proportional to:

```math
P(x,y)
\propto
\left(
I(x,y)+0.035
\right)^{1.5}
```
This gives high-importance locations a greater chance of becoming polygon vertices while still allowing sampling in low-detail regions.

---

## 8. Background Calmness

Background calmness controls how evenly points are distributed in low-detail regions.

For a requested background sample count $N$, an approximate spatial step is:

```math
s
=
\sqrt{
\frac{WH}{N}
}
```
If the calmness parameter is $c\in[0,1]$, sampling jitter is:

```math
J
=
s(0.42-0.32c)
```
Higher calmness therefore reduces random displacement and produces more stable, evenly spaced background cells.

Candidate points are scored using:

```math
S
=
I(x,y)(1.35+0.65c)
+
B(0.35+0.35c)
```
where $B$ is the candidate's normalised displacement from the centre of its sampling cell.

Remaining points favour low-importance regions using:

```math
P_{\text{background}}(x,y)
\propto
\left(
1-I(x,y)+0.03
\right)^{1.30+1.40c}
```
---

# Geometry

## Delaunay Triangulation

Triangle mode builds a Delaunay triangulation from the sampled points using OpenCV `Subdiv2D`.

Conceptually, a Delaunay triangulation tends to avoid extremely narrow triangles by favouring triangulations whose circumcircles contain no other sample sites.

This mode produces the classic triangular Low Poly appearance.

---

## Voronoi Tessellation

For sites:

```math
P=\{p_1,p_2,\ldots,p_n\}
```
the Voronoi region associated with $p_i$ is:

```math
V_i
=
\left\{
x
\mid
\|x-p_i\|
\le
\|x-p_j\|,
\forall j\ne i
\right\}
```
Every point in $V_i$ is therefore at least as close to site $p_i$ as it is to any other site.

This produces irregular polygon cells rather than triangles.

---

# Lloyd Relaxation

Voronoi mode optionally applies Lloyd relaxation.

For each Voronoi region $V_i$, its centroid is computed from image moments:

```math
c_x
=
\frac{M_{10}}{M_{00}}
```
```math
c_y
=
\frac{M_{01}}{M_{00}}
```
Each non-border site is then moved toward its Voronoi centroid:

```math
p_i^{(t+1)}
=
\mathrm{centroid}
\left(
V_i^{(t)}
\right)
```
Repeated iterations produce more regular and visually balanced cells.

Border sites remain fixed so that the tessellation continues to cover the full image.

---

# Polygon Colour Estimation

The original image is used only as a colour source.

An optional Gaussian blur produces:

```math
I_{\sigma}
=
\mathcal{G}_{\sigma}(I)
```
For polygon $P$, its colour is the mean colour of all source pixels contained inside that polygon:

```math
\mathbf{c}_P
=
\frac{1}{|P|}
\sum_{x\in P}
I_{\sigma}(x)
```
The entire polygon is then filled with this single colour.

The renderer does **not** copy the original image into the output buffer.

Instead, the output canvas starts with the global mean colour and is then covered by polygon fills.

This guarantees that fine original-image detail is not directly leaked into the final result.

---

# Palette Quantisation

When palette reduction is enabled, the rendered Low Poly image is clustered using K-means.

For $K$ colour centres $\mu_k$, K-means minimises:

```math
\min_{\{\mu_k\}}
\sum_i
\left\|
x_i-\mu_{z_i}
\right\|^2
```
where:

- $x_i$ is a rendered pixel colour
- $z_i$ is its assigned colour cluster
- $K$ is the selected palette size

The clustering stage operates on the already-rendered Low Poly image rather than blending source-image pixels back into the result.

The OpenCV random generator is explicitly seeded, making palette generation reproducible.

Set **Palette colours = 0** to disable palette quantisation.

---

# Final Softening

Final softening is applied only to the generated Low Poly output.

Let:

- $L$ be the rendered Low Poly image
- $\mathcal{G}_{\sigma}(L)$ be its Gaussian-blurred version
- $\alpha\in[0,1]$ be the final softness strength

The final result is:

```math
L_{\text{final}}
=
(1-\alpha)L
+
\alpha\mathcal{G}_{\sigma}(L)
```
No original-image pixels are introduced during this step.

---

# Parameters

| Parameter | Purpose |
|---|---|
| **Polygon density** | Controls the overall number of importance-weighted sampling points |
| **Edge detail** | Controls the number of sampled Canny edge points |
| **Corner detail** | Controls Shi–Tomasi corner sampling |
| **Background density** | Controls additional points in background regions |
| **Background calmness** | Makes background sampling more regular and low-detail-biased |
| **Texture influence** | Controls the contribution of local texture to the importance map |
| **Urban straight-line enhancement** | Enables Hough line sampling for architectural structures |
| **Structure strength** | Controls the amount of line-based structure sampling |
| **Polygon colour softness** | Blurs the colour source before polygon mean-colour estimation |
| **Final softness** | Blends the Low Poly output with a blurred copy of itself |
| **Final blur radius** | Gaussian blur radius used for final softening |
| **Palette colours** | Optional number of K-means colour clusters; `0` disables quantisation |
| **Voronoi relaxation** | Number of Lloyd-relaxation iterations |
| **Seed** | Controls reproducible random sampling |
| **Preview max side** | Maximum dimension used for preview rendering |

---

# Output Design Principle

The central design constraint of LowPoly Studio is:

> **The final image must remain entirely polygon-generated.**

The source photograph may be used for:

- feature analysis
- gradients
- edges
- texture estimation
- point placement
- polygon colour estimation

It is never blended directly back into the final output.

---

# Project Structure

A minimal repository can use:

```text
LowPoly-Studio/
├── lowpoly_studio.py
├── README.md
├── requirements.txt
├── .gitignore
└── examples/
```

Recommended `requirements.txt`:

```text
opencv-python
numpy
Pillow
```

---

# Platform Support

The application is designed to run on:

- Windows
- macOS
- Linux with Tkinter installed

The interface uses native Tk/ttk widgets, so visual appearance may vary slightly between operating systems.

---

# Notes

- Very high polygon densities or multiple Lloyd-relaxation iterations increase processing time.
- Full-resolution export is more computationally expensive than preview rendering.
- Voronoi relaxation affects only Voronoi mode.
- Palette quantisation is optional and disabled when the palette size is set to `0`.
- The export percentage represents algorithmic stage progress rather than a prediction of remaining wall-clock time.

---

## Author

**Qinghuan-W**

GitHub: [Qinghuan-W](https://github.com/Qinghuan-W)
