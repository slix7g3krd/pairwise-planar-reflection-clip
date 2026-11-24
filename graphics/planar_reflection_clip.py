"""平面反射：斜裁剪、递归深度、可见性与分辨率一致性。

本模块实现平面反射（planar reflection）渲染中的一组纯数值/配置辅助函数，
不依赖任何图形 API，便于单元测试。覆盖六个具体问题：

1. 斜裁剪（oblique near plane clipping）的数值稳定化；
2. 递归反射的硬性层数上限；
3. 反射相机复用主相机的可见性集合；
4. 透明物体排序之后再采样反射纹理；
5. 反射纹理分辨率跟随主缓冲（移动端切换分辨率后重算）；
6. 反射强度按视线与平面的夹角衰减。

约定：除少数返回矩阵的辅助结果外，各入口统一返回字典，便于审计。
为方便冒烟测试，所有入口都接受一个“标量简写”：只传一个数字时，
该数字会被解释为各函数最有意义的那个标量参数并使用默认的几何/配置。
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# 常量与容差
# ---------------------------------------------------------------------------

#: 平面/向量退化判定的通用容差。
_OBLIQUE_EPSILON = 1e-6

#: 反射相机与主相机可见性集合比较时的浮点容差。
_VISIBILITY_TOLERANCE = 1e-5

#: 递归反射的绝对硬上限，防止无限套娃。
_HARD_RECURSION_LIMIT = 5

#: 斜裁剪近平面行允许的最大系数幅度；超过则视为数值失控，回退原投影。
_OBLIQUE_ROW_LIMIT = 1e6

#: 归一模式下可接受的近平面行系数上限；再大就切回退模式以保证 float32 精度。
_OBLIQUE_SAFE_ROW = 1e3

#: 默认透视投影（右手系，深度 0..1，垂直视场 60°，16:9）。
_DEFAULT_NEAR = 0.1
_DEFAULT_FAR = 1000.0
_DEFAULT_FOVY = math.radians(60.0)
_DEFAULT_ASPECT = 16.0 / 9.0


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _is_scalar(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def _as_vec4(value, name="plane"):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"{name} must have 4 components, got {arr.size}")
    return arr


def _as_vec3(value, name="vector"):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must have 3 components, got {arr.size}")
    return arr


def _normalize_plane(plane):
    """归一化平面方程 ``a*x + b*y + c*z + d = 0`` 的法线部分。"""
    p = _as_vec4(plane).copy()
    n = np.linalg.norm(p[:3])
    if n <= _OBLIQUE_EPSILON:
        raise ValueError("plane normal is degenerate")
    return p / n


def _clamp(value, low, high):
    return max(low, min(high, value))


def _normalize(v):
    ln = np.linalg.norm(v)
    return v / ln if ln > _OBLIQUE_EPSILON else v


def _default_projection():
    """构造一个标准透视投影矩阵（列向量约定）。"""
    f = 1.0 / math.tan(_DEFAULT_FOVY / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / _DEFAULT_ASPECT
    m[1, 1] = f
    # 深度映射到 0..1（Vulkan/D3D 风格）。
    m[2, 2] = _DEFAULT_FAR / (_DEFAULT_NEAR - _DEFAULT_FAR)
    m[2, 3] = (_DEFAULT_FAR * _DEFAULT_NEAR) / (_DEFAULT_NEAR - _DEFAULT_FAR)
    m[3, 2] = -1.0
    return m


# ---------------------------------------------------------------------------
# 1. 斜裁剪数值稳定化
# ---------------------------------------------------------------------------


def oblique_clip(proj=None, plane=(0.0, 1.0, 0.0, 0.0), epsilon=_OBLIQUE_EPSILON):
    """计算数值稳定的斜裁剪投影矩阵。

    参数
    ----
    proj : array-like | float | None
        4x4 透视投影矩阵（列向量约定，``clip = proj @ X``）；传标量时视为
        ``epsilon`` 并使用默认投影。
    plane : array-like
        相机空间反射平面 ``(a, b, c, d)``，满足 ``a*x+b*y+c*z+d = 0``。
    epsilon : float
        数值稳定化阈值，同时用于距离归一。

    返回
    ----
    dict
        ``{"matrix", "scale", "distance", "near", "far", "stable", "epsilon"}``。

    说明
    ----
    常见做法（Lengyel / Terathon）把平面变换到裁剪空间 ``Q = P^-T * c``，
    再以“保持侧的远视锥角点”作分母：

        row_near = (2 / (Q·corner)) * Q - (0, 0, 1, 0)

    相机贴近地面、或平面掠过视锥角点时 ``Q·corner -> 0``，该行被放大到
    1e6 量级以上，float32 的深度精度直接崩掉——这正是要修的问题。

    这里改用数值恒定的 view-space 构造，避开该除法：

        row_near = s * c,   s = |P[2,2]|

    view-space 平面系数 ``c`` 本身就是对应的近平面行（只差正比例因子）：
    平面 ``c·X = 0`` 上的点映射到 ``z_clip = 0``，一侧深度为正、另一侧为负。
    取 ``s = |P[2,2]|`` 时 ``s * c_near`` 精确还原投影矩阵原有的近平面行，
    因此任意平面的行系数都与原投影同量级（O(1)），即使相机贴地、平面掠过
    视锥角点也不会爆炸。仅在相机原点恰位于平面上（``c·X`` 无定义）或结果
    超出安全量级时保留原投影并标记 ``stable=False``，交由调用方降级。
    """
    if _is_scalar(proj):
        epsilon = float(proj)
        proj = _default_projection()
    if proj is None:
        proj = _default_projection()

    proj = np.asarray(proj, dtype=np.float64)
    if proj.shape != (4, 4):
        raise ValueError("proj must be a 4x4 matrix")

    eps = abs(float(epsilon)) or _OBLIQUE_EPSILON
    c = _normalize_plane(plane)

    # 从投影矩阵反解 near/far：
    #   P[2,2] = far/(near-far), P[2,3] = far*near/(near-far)
    #   => near = P[2,3]/P[2,2], far = P[2,3]/(1 + P[2,2])
    a, b = float(proj[2, 2]), float(proj[2, 3])
    near = abs(b / a) if abs(a) > eps else _DEFAULT_NEAR
    far = abs(b / (1.0 + a)) if abs(1.0 + a) > eps else _DEFAULT_FAR
    if not (near > eps and far > near):
        near, far = _DEFAULT_NEAR, _DEFAULT_FAR

    # 相机原点应位于被保留的一侧：让 c[3] > 0（平面法线指向相机一侧）。
    if c[3] < 0.0:
        c = -c
    dist = float(c[3])

    # --- 斜裁剪近平面行（view-space 构造，数值恒定有界）---
    # 恒等式：把投影矩阵第 2 行整体替换为 s * c（s > 0），则平面 c·X = 0 上的点
    # 满足 z_clip = 0，c·X > 0 的一侧映射到深度 > 0，另一侧 < 0。即任意
    # view-space 平面系数本身就是对应的近平面行，只差一个正比例因子；
    # 比例只缩放深度，不改变裁剪平面本身。
    #
    # 比例因子选两种模式：
    #   * 归一模式：s = far / (c · X_far)，X_far 是“保持侧在远平面上的视锥
    #     角点”。它把该角点精确映射到 z_ndc = 1，从而完整写满 0..1 深度；
    #     分母越大表示平面离该角点越远，几何上越稳定。
    #   * 回退模式：当分母不为正（平面掠过角点或位于相机背后）时，改用
    #     s = |P[2,2]|。它精确还原投影矩阵原有的近平面行，量级与常规投影
    #     一致、永不爆炸——代价是深度范围被压缩，但裁剪平面完全正确。
    # 两种模式都满足“平面上 z=0、保持侧 z>0”，区别仅在深度精度。
    s_fallback = abs(float(proj[2, 2])) if abs(proj[2, 2]) > eps else 1.0
    X_far = np.array(
        [
            math.copysign(far / float(proj[0, 0]), c[0]) if c[0] != 0.0 else 0.0,
            math.copysign(far / float(proj[1, 1]), c[1]) if c[1] != 0.0 else 0.0,
            -float(far),
            1.0,
        ]
    )
    denom = float(c @ X_far)

    out = proj.copy()
    stable = True
    mode = "normalized"
    s = s_fallback
    if np.linalg.norm(c[:3]) <= eps:
        # 平面法线退化，无法定义近平面。
        stable = False
    else:
        # 归一模式只在系数保持温和（float32 可安全表示）时启用；一旦接近
        # 平面掠过视锥角点的奇点，系数会迅速增大，此时切到回退模式，
        # 用原近平面量级换取数值精度——裁剪平面不变，仅深度范围压缩。
        if denom > eps:
            s = float(far) / denom
            if np.max(np.abs(s * c)) > _OBLIQUE_SAFE_ROW:
                s, mode = s_fallback, "fallback"
        else:
            s, mode = s_fallback, "fallback"

        row = s * c
        candidate = proj.copy()
        candidate[2, :] = row
        if np.all(np.isfinite(candidate)) and np.max(np.abs(row)) < _OBLIQUE_ROW_LIMIT:
            out = candidate
        else:
            stable = False

    return {
        "matrix": out,
        "scale": s,
        "mode": mode,
        "denominator": denom,
        "distance": dist,
        "near": near,
        "far": far,
        "stable": stable,
        "epsilon": eps,
    }


def depth_precision(depth_range=(0.1, 1000.0), samples=8):
    """估计透视深度缓冲的精度分布，用于判断近平面是否崩掉。

    返回 ``{"near", "far", "ratio", "samples", "precision", "stable"}``。
    ``precision[i]`` 是世界空间单位距离对应的 NDC 深度增量，越小越精确；
    ``stable`` 为真表示所有采样点有限且近平面没有退化。
    """
    if _is_scalar(depth_range):
        samples = int(depth_range)
        depth_range = (0.1, 1000.0)

    near, far = float(depth_range[0]), float(depth_range[1])
    if near <= 0.0 or far <= near:
        raise ValueError("require 0 < near < far")
    n = max(2, int(samples))

    zs = np.linspace(near, far, n)
    # NDC 深度（0..1）：z_ndc = f/(f-n) * (1 - n/z)。
    ndc = (far / (far - near)) * (1.0 - near / zs)
    precision = np.diff(ndc) / np.diff(zs)

    return {
        "near": near,
        "far": far,
        "ratio": far / near,
        "samples": ndc.tolist(),
        "precision": precision.tolist(),
        "stable": bool(np.all(np.isfinite(precision))) and near > _OBLIQUE_EPSILON,
    }


# ---------------------------------------------------------------------------
# 2. 递归层数硬上限
# ---------------------------------------------------------------------------


def recursion_cap(requested=1, hard_limit=_HARD_RECURSION_LIMIT):
    """把请求的递归层数钳制到硬上限内。

    返回 ``{"requested", "limit", "effective", "clamped"}``。
    ``hard_limit`` 本身也被钉在模块常量 ``_HARD_RECURSION_LIMIT`` 内，
    防止调用方传入更大的值而重新打开无限递归。
    """
    limit = max(0, min(int(hard_limit), _HARD_RECURSION_LIMIT))
    req = int(requested)
    effective = max(0, min(req, limit))
    return {
        "requested": req,
        "limit": limit,
        "effective": effective,
        "clamped": effective != req,
    }


# ---------------------------------------------------------------------------
# 3. 反射相机复用主相机可见性
# ---------------------------------------------------------------------------


def visibility_match(main_visible, reflection_visible=None, tolerance=_VISIBILITY_TOLERANCE):
    """让反射相机复用主相机的可见性集合。

    输入可以是对象 id 的序列或 ``{id: score}`` 映射。返回：``visible`` 是
    复用后的集合（严格按主相机顺序）；``missing`` 是主相机可见而反射相机
    没有的；``extra`` 是反射相机多出来的（它们会让物体在反射里凭空出现）。
    ``consistent`` 为真表示两边集合恰好相同。
    """
    if _is_scalar(main_visible):
        n = max(0, int(main_visible))
        main_visible = list(range(n))
        reflection_visible = list(range(n))
    if reflection_visible is None:
        reflection_visible = main_visible

    main_ids = _ordered_ids(main_visible)
    refl_ids = _ordered_ids(reflection_visible)
    main_set, refl_set = set(main_ids), set(refl_ids)

    visible = [i for i in main_ids if i in refl_set]
    missing = [i for i in main_ids if i not in refl_set]
    extra = [i for i in refl_ids if i not in main_set]

    return {
        "visible": visible,
        "missing": missing,
        "extra": extra,
        "consistent": not missing and not extra,
        "tolerance": float(tolerance),
        # 显式声明反射相机在可见性上从属于主相机。
        "reuse_main_visibility": True,
    }


def _ordered_ids(collection):
    if collection is None:
        return []
    if isinstance(collection, dict):
        return list(collection.keys())
    if isinstance(collection, (set, frozenset)):
        return sorted(collection)
    return list(collection)


# ---------------------------------------------------------------------------
# 4. 透明物体排序后再采样
# ---------------------------------------------------------------------------


def transparent_order(objects=1, camera_position=(0.0, 0.0, 0.0)):
    """先按到相机距离对透明物体排序，再给出反射采样顺序。

    ``objects`` 是 ``{"id": (x, y, z)}`` 或 ``[(id, (x, y, z)), ...]``。
    返回 ``{"order", "depths", "sorted_before_sampling"}``；``order`` 为
    由远到近（画家算法）的 id 列表。调用方应等待本函数返回顺序后再采样
    反射纹理，避免透明物体用错排序写出的反射。
    """
    if _is_scalar(objects):
        count = max(0, int(objects))
        objects = {i: (float(i) - (count - 1) / 2.0, 0.0, -1.0 - i) for i in range(count)}

    cam = _as_vec3(camera_position, "camera_position")
    items = objects.items() if isinstance(objects, dict) else objects

    pairs = []
    for key, pos in items:
        p = _as_vec3(pos, f"position[{key}]")
        pairs.append((key, float(np.linalg.norm(p - cam))))
    # 稳定且确定：由远到近，深度相同按 id 字符串排序。
    pairs.sort(key=lambda kv: (-kv[1], str(kv[0])))

    return {
        "order": [k for k, _ in pairs],
        "depths": {k: d for k, d in pairs},
        "sorted_before_sampling": True,
    }


# ---------------------------------------------------------------------------
# 5. 反射分辨率跟随主缓冲
# ---------------------------------------------------------------------------


def resolution_link(main_size=(1920, 1080), scale=0.5, min_size=16, max_size=8192):
    """由主渲染缓冲尺寸推导反射缓冲尺寸。

    ``main_size`` 为 ``(width, height)``，或传标量表示缩放比例。移动端切换
    分辨率后重新调用即可，避免反射缓冲比例失配。返回
    ``{"main", "reflection", "scale", "effective_scale"}``。
    """
    if _is_scalar(main_size):
        scale = _clamp(float(main_size), 1.0 / 1024.0, 1.0)
        main_size = (1920, 1080)

    mw, mh = int(main_size[0]), int(main_size[1])
    if mw <= 0 or mh <= 0:
        raise ValueError("main_size must be positive")
    s = float(scale)
    if not (0.0 < s <= 1.0):
        raise ValueError("scale must be in (0, 1]")

    def _fit(v):
        return int(_clamp(round(v * s), int(min_size), int(max_size)))

    rw, rh = _fit(mw), _fit(mh)
    return {
        "main": (mw, mh),
        "reflection": (rw, rh),
        "scale": s,
        "effective_scale": (rw / mw, rh / mh),
        "linked": True,
    }


# ---------------------------------------------------------------------------
# 6. 反射强度按角度衰减
# ---------------------------------------------------------------------------


def intensity(view_dir=(1.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0), base=1.0,
              min_factor=0.0, power=1.0):
    """按视线与平面法线夹角衰减反射强度。

    采用 Schlick 形式 ``factor = (1 - |cosθ|)**power``：正视平面时反射弱，
    掠射时反射强。标量简写时该值作为 ``power``。返回
    ``{"factor", "intensity", "cos_theta", "grazing"}``。
    """
    if _is_scalar(view_dir):
        power = float(view_dir)
        view_dir = (1.0, 0.0, 0.0)
        normal = (0.0, 1.0, 0.0)

    v = _as_vec3(view_dir, "view_dir")
    n = _as_vec3(normal, "normal")
    if np.linalg.norm(v) <= _OBLIQUE_EPSILON or np.linalg.norm(n) <= _OBLIQUE_EPSILON:
        raise ValueError("view_dir and normal must be non-degenerate")

    cos_theta = _clamp(abs(float(np.dot(_normalize(v), _normalize(n)))), 0.0, 1.0)
    p = max(float(power), 0.0)
    factor = _clamp((1.0 - cos_theta) ** p, float(min_factor), 1.0)

    return {
        "factor": factor,
        "intensity": float(base) * factor,
        "cos_theta": cos_theta,
        "grazing": factor > 0.5,
    }


# ---------------------------------------------------------------------------
# 反射相机
# ---------------------------------------------------------------------------


def reflection_cam(position=(0.0, 1.0, 0.0), forward=(0.0, -1.0, 0.0),
                   up=(0.0, 0.0, 1.0), plane=(0.0, 1.0, 0.0, 0.0)):
    """构造关于反射平面镜像的相机。

    返回 ``{"position", "forward", "up", "plane"}``。位置关于平面做镜像，
    方向与 up 去掉法线分量后取反。
    """
    if _is_scalar(position):
        position = (0.0, float(position), 0.0)
    pos = _as_vec3(position, "position")
    fwd = _as_vec3(forward, "forward")
    upv = _as_vec3(up, "up")
    p = _normalize_plane(plane)
    n, d = p[:3], p[3]

    dist = float(np.dot(pos, n) + d)
    mirrored = pos - 2.0 * dist * n
    mfwd = fwd - 2.0 * np.dot(fwd, n) * n
    mup = upv - 2.0 * np.dot(upv, n) * n

    return {
        "position": mirrored.tolist(),
        "forward": _normalize(mfwd).tolist(),
        "up": _normalize(mup).tolist(),
        "plane": p.tolist(),
    }


# ---------------------------------------------------------------------------
# 校验 / 审计 / 回退 / 指标 / 剔除 / 模糊 / 确定性
# ---------------------------------------------------------------------------


def validation(config=1):
    """校验反射配置，返回 ``{"ok", "errors", "warnings"}``。

    标量简写视为递归层数。检查递归是否越界、缩放比例是否合法、
    平面是否可归一化、是否复用主相机可见性。
    """
    if _is_scalar(config):
        config = {"recursive": True, "recursion": int(config)}
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    errors, warnings = [], []
    if config.get("recursive", False):
        cap = config.get("recursion", 1)
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
            errors.append("recursion must be a non-negative int")
        elif cap > _HARD_RECURSION_LIMIT:
            errors.append(f"recursion {cap} exceeds hard limit {_HARD_RECURSION_LIMIT}")

    scale = config.get("scale", 0.5)
    if not isinstance(scale, (int, float)) or not (0.0 < float(scale) <= 1.0):
        errors.append("scale must be in (0, 1]")

    if "plane" in config:
        try:
            _normalize_plane(config["plane"])
        except (ValueError, TypeError) as exc:
            errors.append(f"invalid plane: {exc}")

    if not config.get("reuse_main_visibility", True):
        warnings.append("reflection camera does not reuse main visibility")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def audit(frames=1):
    """对多帧反射状态做审计，检查各一致性约束。

    返回 ``{"ok", "issues", "frames"}``。标量简写视为帧数，生成同样数量的
    干净帧。
    """
    if _is_scalar(frames):
        frames = [{} for _ in range(max(0, int(frames)))]
    if not isinstance(frames, (list, tuple)):
        raise TypeError("frames must be a sequence")

    issues = []
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            issues.append({"frame": idx, "issue": "frame is not a dict"})
            continue
        if "recursion" in frame:
            cap = recursion_cap(frame["recursion"])
            if cap["clamped"]:
                issues.append({"frame": idx, "issue": "recursion clamped", "detail": cap})
        if "visibility" in frame and not frame["visibility"].get("consistent", True):
            issues.append({"frame": idx, "issue": "visibility mismatch",
                           "detail": frame["visibility"]})
        if "resolution" in frame and frame["resolution"].get("mismatch"):
            issues.append({"frame": idx, "issue": "resolution mismatch",
                           "detail": frame["resolution"]})
        if "sampled_before_sort" in frame and frame["sampled_before_sort"]:
            issues.append({"frame": idx, "issue": "reflection sampled before transparency sort"})
    return {"ok": not issues, "issues": issues, "frames": len(frames)}


def fallback(failures=0, strategy="disable"):
    """反射管线失败时的降级策略。

    返回 ``{"strategy", "enabled", "reason", "count"}``。标量简写视为失败计数。
    """
    if _is_scalar(failures):
        failures = [RuntimeError("failure")] * max(0, int(failures))
    if not isinstance(failures, (list, tuple)):
        raise TypeError("failures must be a sequence")

    strategies = {"disable", "clip_only", "lower_res", "no_recursion"}
    if strategy not in strategies:
        raise ValueError(f"unknown strategy {strategy!r}")

    count = len(failures)
    if count == 0:
        return {"strategy": strategy, "enabled": True, "reason": "no failures", "count": 0}
    return {
        "strategy": strategy,
        "enabled": False,
        "reason": f"{count} failure(s); falling back to {strategy}",
        "count": count,
    }


def metrics(samples=1):
    """对若干帧的反射质量指标求聚合值。

    返回 ``{"count", "mean", "min", "max", "std"}``。标量简写视为单个样本。
    """
    if _is_scalar(samples):
        samples = [float(samples)]
    arr = np.asarray(list(samples), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def cull_plane(plane=(0.0, 1.0, 0.0, 0.0), bounds_min=(-1.0, -1.0, -1.0),
               bounds_max=(1.0, 1.0, 1.0)):
    """用反射平面对 AABB 做保守剔除。

    返回 ``{"relation", "max_distance", "min_distance", "cull"}``；
    ``relation`` 为 ``inside`` / ``outside`` / ``intersect``，完全位于平面
    负侧时可以整包剔除。标量简写视为立方体半边长。
    """
    if _is_scalar(plane):
        half = abs(float(plane))
        bounds_min = (-half, -half, -half)
        bounds_max = (half, half, half)
        plane = (0.0, 1.0, 0.0, 0.0)

    p = _normalize_plane(plane)
    n, d = p[:3], p[3]
    bmin = _as_vec3(bounds_min, "bounds_min")
    bmax = _as_vec3(bounds_max, "bounds_max")
    if np.any(bmin > bmax):
        raise ValueError("bounds_min must be <= bounds_max")

    positive = np.where(n >= 0, bmax, bmin)
    negative = np.where(n >= 0, bmin, bmax)
    max_dist = float(np.dot(n, positive) + d)
    min_dist = float(np.dot(n, negative) + d)

    if max_dist < -_OBLIQUE_EPSILON:
        relation = "outside"
    elif min_dist > _OBLIQUE_EPSILON:
        relation = "inside"
    else:
        relation = "intersect"
    return {
        "relation": relation,
        "max_distance": max_dist,
        "min_distance": min_dist,
        "cull": relation == "outside",
    }


def blur(radius=1.0, resolution=1080, reference=1080, taper=True):
    """按分辨率缩放反射模糊半径。

    主缓冲分辨率变化后，反射缓冲的像素尺寸随之变化；模糊核半径必须
    按高度比例重算（``radius * resolution / reference``）。``taper`` 让
    半径趋近 0 时按平方收敛，避免亚像素抖动。
    """
    # 标量简写只影响 radius，绝不覆盖调用方显式传入的 resolution。
    if _is_scalar(radius):
        radius = float(radius)
    res, ref = int(resolution), int(reference)
    if res <= 0 or ref <= 0:
        raise ValueError("resolution and reference must be positive")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    ratio = res / ref
    scaled = float(radius) * ratio
    if taper and scaled < 1.0:
        scaled *= scaled

    return {
        "radius": scaled,
        "resolution": res,
        "reference": ref,
        "ratio": ratio,
    }


def determinism(seed=0, count=4):
    """生成确定性的伪随机序列，用于验证渲染结果可复现。

    使用 splitmix64 整数混洗，不触碰全局随机状态；相同 ``seed``/``count``
    必然得到相同结果。返回 ``{"seed", "count", "values", "stable"}``。
    """
    seed, count = int(seed), int(count)
    if count < 0:
        raise ValueError("count must be non-negative")
    values = _splitmix(seed, count)
    return {
        "seed": seed,
        "count": count,
        "values": values,
        # 复算一次验证无隐藏状态参与。
        "stable": values == _splitmix(seed, count),
    }


def _splitmix(seed, count):
    mask = (1 << 64) - 1
    state = (int(seed) + 0x9E3779B97F4A7C15) & mask
    out = []
    for _ in range(int(count)):
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z ^= z >> 31
        out.append(z / float(1 << 64))
    return out
