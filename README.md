# LowPoly Studio

[中文](README.md) | [English](README_EN.md)

LowPoly Studio 是一个桌面端图像风格化工具，可将普通照片转换为 **100% 由多边形生成的 Low Poly 艺术图像**。

程序支持两种几何模式：

- **Triangle Low Poly（三角 Low Poly）** —— 基于内容感知采样的 Delaunay 三角剖分
- **Voronoi Polygon（Voronoi 多边形）** —— Voronoi 泰森多边形，并支持可选的 Lloyd 松弛

与将原图重新叠加回最终图像的方法不同，LowPoly Studio 只使用原始图像进行 **特征分析与多边形颜色估计**。最终输出图像完全由多边形绘制得到。

---

## 功能特性

- **100% 多边形输出**
  - 最终结果不会重新混入原始图像。
  - 极少数未被多边形覆盖的像素使用全局平均颜色填充，而不是使用原图像素。

- **两种几何模式**
  - Delaunay 三角网格
  - Voronoi 多边形网格

- **内容感知采样**
  - 灰度梯度
  - 颜色梯度
  - Canny 边缘
  - 局部纹理
  - Shi–Tomasi 角点检测
  - 可选 Hough 直线采样，用于增强建筑与城市结构

- **Voronoi Lloyd 松弛**
  - 使 Voronoi 单元分布更加均匀。
  - 边界采样点保持固定，从而保证图像完整覆盖。

- **可调节艺术参数**
  - 多边形密度
  - 边缘细节
  - 角点细节
  - 背景密度
  - 背景宁静度
  - 纹理影响
  - 结构增强
  - 多边形取色柔化
  - 最终柔化
  - 色板颜色数
  - Voronoi 松弛次数
  - 随机种子

- **可复现生成**
  - NumPy 与 OpenCV 的随机数生成器均使用 Seed 控制。
  - 相同图片、相同参数与相同 Seed 可以得到一致结果。

- **预览与原分辨率导出**
  - 预览时可降低处理分辨率以提升速度。
  - PNG 导出时使用原始图片分辨率重新生成。

- **导出进度弹窗**
  - 提供 0% 到 100% 的可视化导出进度。

- **中英文界面**
  - 中文
  - English

---

## 环境要求

推荐：

- Python 3.10+
- OpenCV
- NumPy
- Pillow
- Tkinter

安装 Python 依赖：

```bash
pip install opencv-python numpy Pillow
```

Tkinter 通常已经包含在标准 Python 发行版中。

Linux 环境如果没有 Tkinter，可单独安装，例如：

```bash
sudo apt install python3-tk
```

---

## 运行方式

克隆仓库：

```bash
git clone https://github.com/Qinghuan-W/LowPoly-Studio.git
cd LowPoly-Studio
```

运行程序：

```bash
python lowpoly_studio.py
```

部分系统需要显式使用 Python 3：

```bash
python3 lowpoly_studio.py
```

---

# 算法流程

LowPoly Studio 的整体处理流程如下：

```text
输入图像
    │
    ▼
双边滤波平滑
    │
    ▼
图像特征分析
 ├─ 灰度梯度
 ├─ 颜色梯度
 ├─ Canny 边缘图
 └─ 局部纹理图
    │
    ▼
Importance Map（重要性图）
    │
    ▼
内容感知采样
 ├─ 图像边界点
 ├─ Shi–Tomasi 角点
 ├─ 边缘采样点
 ├─ 可选 Hough 直线采样点
 ├─ Importance 加权采样点
 └─ 背景平静区域采样点
    │
    ▼
采样点去重
    │
    ├───────────────┐
    ▼               ▼
Delaunay         Voronoi
三角剖分          多边形划分
                    │
                    ▼
               Lloyd 松弛
                （可选）
    │               │
    └───────┬───────┘
            ▼
      多边形区域平均取色
            │
            ▼
      可选 K-means 色板量化
            │
            ▼
      可选 Low-Poly-only 柔化
            │
            ▼
          PNG 输出
```

---

# 数学模型

## 1. 特征图

设输入图像为：

```math
I(x,y)
```
在提取特征之前，程序首先使用双边滤波进行平滑，在降低弱噪声的同时尽可能保留较明显的边缘结构。

---

## 2. 灰度梯度

设灰度图像为 $Y$，使用 Sobel 算子分别计算水平方向与垂直方向梯度：

```math
G_x = S_x * Y
```
```math
G_y = S_y * Y
```
梯度幅值为：

```math
G_{\text{gray}}(x,y)
=
\sqrt{
G_x(x,y)^2 + G_y(x,y)^2
}
```
计算结果随后进行 Gaussian 平滑，并归一化到：

```math
[0,1]
```
---

## 3. 颜色梯度

程序首先将图像由 BGR 转换到 CIELAB 色彩空间。

对于每一个 LAB 通道 $k$，分别计算：

```math
G_k(x,y)
=
\sqrt{
G_{x,k}(x,y)^2
+
G_{y,k}(x,y)^2
}
```
然后将各通道的梯度响应相加：

```math
G_{\text{colour}}(x,y)
=
\mathrm{Norm}
\left(
\sum_k G_k(x,y)
\right)
```
使用 LAB 而不是直接在 BGR 空间中计算，是因为 LAB 能够更明确地区分亮度信息和颜色信息。

---

## 4. 局部纹理

局部纹理由灰度图像的局部方差估计。

首先计算局部均值：

```math
\mu(x,y)
=
\mathcal{G}_{\sigma}(Y)
```
再计算灰度平方后的局部均值：

```math
\mu_2(x,y)
=
\mathcal{G}_{\sigma}(Y^2)
```
局部方差为：

```math
V(x,y)
=
\max
\left(
\mu_2(x,y)-\mu(x,y)^2,
0
\right)
```
随后再次平滑并进行归一化：

```math
T(x,y)
=
\mathrm{Norm}(V(x,y))
```
其中 $T$ 表示最终的纹理响应图。

---

## 5. Importance Map

LowPoly Studio 将灰度结构、颜色结构、纹理与 Canny 边缘组合为统一的重要性图。

当前实现为：

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
其中：

- $G_{\text{gray}}$：归一化后的灰度梯度
- $G_{\text{colour}}$：归一化后的 LAB 颜色梯度
- $T$：纹理图
- $w_t$：用户设置的纹理影响参数
- $E$：膨胀后的 Canny 边缘图

最终的重要性图为：

```math
I
=
\mathrm{Norm}
\left(
\mathcal{G}(I_{\text{raw}})
\right)
```
因此，视觉结构越明显的位置会获得越高的重要性值，并在后续采样中获得更多采样点。

---

## 6. 图像复杂度估计

程序根据边缘密度和平均 Importance 值估计当前图片的复杂度：

```math
C_{\text{raw}}
=
0.65
+
3.2d_e
+
1.35\bar{I}
```
其中：

```math
d_e
=
\frac{\text{边缘像素数量}}
{\text{图像总像素数量}}
```
以及：

```math
\bar{I}
=
\frac{1}{WH}
\sum_{x,y}I(x,y)
```
为了避免极端值，复杂度最终限制为：

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
这样可以让复杂图像自动获得更多几何采样，而简单图像不会被过度细分。

---

## 7. 自适应采样点数量

设图像宽高比修正项为：

```math
A
=
\frac{\max(W,H)}
{\min(W,H)}
```
并定义：

```math
R=\sqrt{A}
```
不同类型的采样点数量近似为：

### Importance 加权采样点

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
### 边缘采样点

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
### 角点采样点

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
### 背景采样点

```math
N_{\text{background}}
=
75
\cdot
R
\cdot
D_b
```
其中：

- $D_p$：多边形密度
- $D_e$：边缘细节
- $D_c$：角点细节
- $D_b$：背景密度

程序还会对这些数量进行限制，以控制运行时间和内存消耗。

---

## 8. Importance 加权采样

Importance Map 决定普通自适应采样点出现的概率。

像素被选中的概率近似满足：

```math
P(x,y)
\propto
\left(
I(x,y)+0.035
\right)^{1.5}
```
因此：

- 高 Importance 区域更容易得到采样点
- 低 Importance 区域仍保留一定采样概率
- 轮廓和结构区域会得到更高几何密度

---

## 9. 背景宁静度

背景宁静度用于控制低细节区域中采样点的规则程度。

对于目标背景点数量 $N$，程序首先计算近似网格间距：

```math
s
=
\sqrt{
\frac{WH}{N}
}
```
若背景宁静度为：

```math
c\in[0,1]
```
则随机偏移范围为：

```math
J
=
s(0.42-0.32c)
```
因此：

- $c$ 越大，随机抖动越小
- 背景点越趋向均匀分布
- 大面积天空、墙面等区域会更加稳定

候选点的评分函数为：

```math
S
=
I(x,y)(1.35+0.65c)
+
B(0.35+0.35c)
```
其中 $B$ 表示候选点偏离网格中心的归一化程度。

对于剩余背景点，采样概率为：

```math
P_{\text{background}}(x,y)
\propto
\left(
1-I(x,y)+0.03
\right)^{1.30+1.40c}
```
因此背景宁静度越高，采样越倾向于低 Importance 区域。

---

# 几何生成

## Delaunay 三角剖分

Triangle 模式使用 OpenCV `Subdiv2D` 根据采样点生成 Delaunay 三角剖分。

Delaunay 三角剖分的经典性质是：

> 对于每一个三角形，其外接圆内部不存在其他采样点。

它通常能够避免大量过于狭长的三角形，因此非常适合经典 Low Poly 风格。

---

## Voronoi 多边形

设采样点集合为：

```math
P=\{p_1,p_2,\ldots,p_n\}
```
点 $p_i$ 对应的 Voronoi 区域定义为：

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
即：

> Voronoi 区域中的任意位置，到当前采样点 $p_i$ 的距离都不大于到其他采样点的距离。

与三角模式相比，Voronoi 可以生成更加自由的不规则多边形块。

---

# Lloyd Relaxation

Voronoi 模式支持 Lloyd 松弛。

对于 Voronoi 区域 $V_i$，通过图像矩计算其质心：

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
随后将对应的采样点移动到 Voronoi 单元质心：

```math
p_i^{(t+1)}
=
\mathrm{centroid}
\left(
V_i^{(t)}
\right)
```
经过多次迭代后，Voronoi 单元的分布会更加均匀和平衡。

程序会固定图像边缘上的采样点，以避免松弛后破坏完整画面覆盖。

---

# 多边形颜色估计

原始图像只作为颜色来源。

首先可以根据用户设置进行 Gaussian Blur：

```math
I_{\sigma}
=
\mathcal{G}_{\sigma}(I)
```
对于多边形区域 $P$，其填充颜色为该区域内部所有源图像像素的平均颜色：

```math
\mathbf{c}_P
=
\frac{1}{|P|}
\sum_{x\in P}
I_{\sigma}(x)
```
然后使用这一颜色将整个多边形进行纯色填充。

程序不会执行：

```python
output = original.copy()
```

而是先创建一个使用全局平均颜色初始化的输出画布，再逐个绘制多边形。

因此最终结果不会直接泄漏原始图像的细节。

---

# 色板量化

当启用色板颜色数时，程序对已经生成的 Low Poly 图像进行 K-means 聚类。

设共有 $K$ 个颜色中心 $\mu_k$，K-means 的优化目标为：

```math
\min_{\{\mu_k\}}
\sum_i
\left\|
x_i-\mu_{z_i}
\right\|^2
```
其中：

- $x_i$：Low Poly 输出图像中的像素颜色
- $z_i$：像素所属的颜色类别
- $K$：用户设置的色板颜色数

需要注意：

> K-means 作用于已经生成的 Low Poly 图像，而不是重新从原图混合像素。

程序同时固定 OpenCV 的随机种子，因此色板聚类结果可以复现。

将：

```text
色板颜色数 = 0
```

即可关闭色板量化。

---

# 最终柔化

最终柔化同样只作用于已经生成的 Low Poly 图像。

设：

- $L$：原始 Low Poly 输出
- $\mathcal{G}_{\sigma}(L)$：Low Poly 输出的 Gaussian Blur 版本
- $\alpha\in[0,1]$：最终柔化强度

最终图像为：

```math
L_{\text{final}}
=
(1-\alpha)L
+
\alpha\mathcal{G}_{\sigma}(L)
```
该步骤不会重新混入原始照片。

---

# 参数说明

| 参数 | 作用 |
|---|---|
| **多边形密度** | 控制 Importance 加权采样点的总体数量 |
| **边缘细节** | 控制 Canny 边缘采样点数量 |
| **角点细节** | 控制 Shi–Tomasi 角点数量 |
| **背景密度** | 控制背景区域额外采样点数量 |
| **背景宁静度** | 使背景采样更均匀，并更加偏向低细节区域 |
| **纹理影响** | 控制局部纹理在 Importance Map 中的权重 |
| **城市直线结构增强** | 使用 HoughLinesP 提取建筑和城市直线结构 |
| **结构增强强度** | 控制直线结构采样强度 |
| **多边形取色柔化** | 在计算多边形平均颜色之前对颜色源进行模糊 |
| **最终柔化强度** | 将 Low Poly 输出与其自身模糊版本进行混合 |
| **最终模糊半径** | 最终柔化所使用的 Gaussian Blur 半径 |
| **色板颜色数** | 可选 K-means 颜色类别数量；`0` 表示关闭 |
| **Voronoi 松弛** | Lloyd Relaxation 的迭代次数 |
| **随机种子** | 控制随机采样与色板聚类，使结果可复现 |
| **预览最长边** | 控制预览渲染时使用的最大图像尺寸 |

---

# 输出设计原则

LowPoly Studio 的核心设计约束是：

> **最终图像必须完全由几何多边形生成。**

原始照片可以用于：

- 图像特征分析
- 梯度计算
- 边缘检测
- 纹理估计
- 采样点布局
- 多边形颜色计算

但不会直接叠加回最终输出。

---

# 项目结构

推荐仓库结构：

```text
LowPoly-Studio/
├── lowpoly_studio.py
├── README.md
├── requirements.txt
├── .gitignore
└── examples/
```

推荐 `requirements.txt`：

```text
opencv-python
numpy
Pillow
```

---

# 平台支持

目前程序设计为可运行于：

- Windows
- macOS
- Linux（需安装 Tkinter）

由于 GUI 使用原生 Tk/ttk 控件，不同操作系统上的控件样式可能略有不同，但不会影响核心功能。

---

# 使用说明

- 极高的多边形密度会明显增加计算时间。
- Lloyd Relaxation 迭代次数越高，Voronoi 渲染耗时越长。
- 原分辨率导出会比预览模式消耗更多计算资源。
- Voronoi 松弛只作用于 Voronoi 模式。
- `色板颜色数 = 0` 时关闭色板量化。
- 导出百分比表示算法阶段进度，并不代表对剩余实际时间的精确预测。

---

## 作者

**Qinghuan-W**

GitHub: [Qinghuan-W](https://github.com/Qinghuan-W)
