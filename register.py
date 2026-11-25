"""通用 PLY 点云 Sim(3) 配准。输出列向量变换：目标点 = s R 源点 + t。"""
import argparse
from itertools import permutations, product
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.spatial.transform import Rotation


def read_xyz(path):
    """按顶点属性名称读取 XYZ，忽略颜色、法线及顶点之后的面数据。"""
    types = dict(char="i1", uchar="u1", short="i2", ushort="u2", int="i4",
                 uint="u4", float="f4", double="f8", int8="i1", uint8="u1",
                 int16="i2", uint16="u2", int32="i4", uint32="u4",
                 float32="f4", float64="f8")
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"{path} 不是 PLY 文件")
        fields = []
        element = None
        count = 0
        encoding = None
        for line in stream:
            words = line.decode("ascii").split()
            if not words:
                continue
            if words[0] == "format":
                encoding = words[1]
            elif words[0] == "element":
                element = words[1]
                if element == "vertex":
                    count = int(words[2])
                elif not count and int(words[2]):
                    raise ValueError("要求 vertex 为首个非空数据元素")
            elif words[0] == "property" and element == "vertex":
                if words[1] not in types:
                    raise ValueError("顶点属性需为标量")
                fields.append((words[2], types[words[1]]))
            elif words[0] == "end_header":
                break
        else:
            raise ValueError("缺少 end_header")
        names = [name for name, _ in fields]
        columns = [names.index(axis) for axis in "xyz"]
        if count < 3:
            raise ValueError("点云至少需要三个有效的非共线点")
        if encoding == "ascii":
            data = np.loadtxt(stream, usecols=columns, max_rows=count, ndmin=2)
            if len(data) != count:
                raise ValueError("顶点数与文件头不一致")
        else:
            endian = {"binary_little_endian": "<", "binary_big_endian": ">"}[encoding]
            dtype = np.dtype([(name, endian + kind) for name, kind in fields])
            records = np.memmap(path, dtype=dtype, mode="r", offset=stream.tell(), shape=(count,))
            data = np.column_stack([records[axis] for axis in "xyz"])
    valid = np.isfinite(data).all(axis=1)
    return data if valid.all() else data[valid]


def frame(points, resolution):
    """使用全部有效点求中心和主轴；体素原点、形状和边长均由本点云确定。"""
    if len(points) < 3:
        raise ValueError("有效点不足")
    center = points.mean(axis=0, dtype=np.float64)
    covariance = np.zeros((3, 3))
    for start in range(0, len(points), 250000):
        delta = points[start:start + 250000] - center
        covariance += delta.T @ delta
    values, axes = np.linalg.eigh(covariance)
    if values[1] <= np.finfo(float).eps * values[2]:
        raise ValueError("点云共线，无法确定完整旋转")
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    low = np.full(3, np.inf)
    high = -low
    def chunks():
        for start in range(0, len(points), 250000):
            yield (points[start:start + 250000] - center) @ axes
    for xyz in chunks():
        low = np.minimum(low, xyz.min(axis=0))
        high = np.maximum(high, xyz.max(axis=0))
    step = (high - low).max() / resolution
    shape = np.floor((high - low) / step).astype(int) + 1
    counts = np.zeros(shape, dtype=np.int64)
    for xyz in chunks():
        idx = np.floor((xyz - low) / step).astype(int)
        np.add.at(counts, tuple(idx.T), 1)
    return center, axes, low, step, counts


def register(source, target, args):
    cs, us, origin, hs, counts = frame(source, args.resolution)
    ct, ut, low, ht, target_counts = frame(target, args.resolution)
    src = (np.argwhere(counts > 0) + 0.5) * hs + origin
    weights = np.sqrt(counts[counts > 0])
    weights /= weights.sum()
    # 使用目标体素为长度单位，便于同一组数值容差适配米、毫米等坐标系。
    src /= hs
    target_origin = low / ht + 0.5
    occupied = target_counts > 0
    distance = ndimage.distance_transform_edt(np.pad(~occupied, 1, constant_values=True))
    proximity = np.exp(-0.5 * distance[1:-1, 1:-1, 1:-1] ** 2)
    reference = ht / hs
    lower = args.scale_min if args.scale_min is not None else reference / 2
    upper = args.scale_max if args.scale_max is not None else reference * 2
    if not 0 < lower <= upper or not np.isfinite(upper):
        raise ValueError("尺度区间必须满足 0 < scale-min <= scale-max")
    scales = np.unique(np.geomspace(lower, upper, args.scale_steps)) / reference
    candidates = []
    # 两套主轴都有符号和轴序歧义。枚举 24 种保向组合，无需预设竖直轴。
    for perm in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rotation = np.eye(3)[list(perm)] * np.array(signs)[:, None]
            if np.linalg.det(rotation) < 0:
                continue
            rotated = src @ rotation.T
            for scale in scales:
                indices = np.rint(rotated * scale).astype(int)
                offset = indices.min(axis=0)
                indices -= offset
                template = np.zeros(indices.max(axis=0) + 1)
                template[tuple(indices.T)] = 1
                # 全三维线性互相关，同时搜索 XYZ 平移。按模板占据数归一化。
                correlation = fftconvolve(proximity, template[::-1, ::-1, ::-1], mode="full")
                peak = np.array(np.unravel_index(correlation.argmax(), correlation.shape))
                shift = target_origin + peak - np.array(template.shape) + 1 - offset
                score = correlation[tuple(peak)] / template.sum()
                candidates.append((score, rotation, scale, shift))
    # 对粗匹配得分最高的不同初值，在同一目标距离场上优化七个自由度。
    results = []
    for _, initial, scale, shift in sorted(candidates, key=lambda c: -c[0])[:args.starts]:
        def transform(x):
            return Rotation.from_rotvec(x[:3]).as_matrix() @ initial
        def loss(x):
            xyz = (src @ transform(x).T) * np.exp(x[3]) + x[4:]
            coords = (xyz - target_origin + 1).T
            residual = ndimage.map_coordinates(distance, coords, order=1, mode="constant", cval=4.)
            return np.dot(weights, np.minimum(residual, 4.))
        start = np.r_[np.zeros(3), np.log(scale), shift]
        bounds = [(-np.pi / 2, np.pi / 2)] * 3 + [(np.log(lower/reference), np.log(upper/reference))]
        bounds += [(v - 5, v + 5) for v in shift]
        fit = minimize(loss, start, method="Powell", bounds=bounds)
        x = fit.x if fit.fun < loss(start) else start
        results.append((loss(x), x.copy(), transform(x)))
    error, x, local_rotation = min(results, key=lambda result: result[0])
    scale = reference * np.exp(x[3])
    rotation = ut @ local_rotation @ us.T
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = ct + ut @ (ht * x[4:]) - matrix[:3, :3] @ cs
    print(f"尺度搜索区间 [{lower:.8g}, {upper:.8g}]，估计尺度 {scale:.10g}")
    print(f"截断加权单向距离 {error * ht:.8g}（目标单位），目标体素边长 {ht:.8g}")
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--output", type=Path, help="显式指定矩阵文件路径")
    parser.add_argument("--resolution", type=int, default=48, help="最长主轴包围盒边的体素数")
    parser.add_argument("--scale-min", type=float, help="尺度下界，目标单位/源单位")
    parser.add_argument("--scale-max", type=float, help="尺度上界，目标单位/源单位")
    parser.add_argument("--scale-steps", type=int, default=7, help="粗搜索的对数等距尺度数")
    parser.add_argument("--starts", type=int, default=5, help="进入局部优化的初值数")
    args = parser.parse_args()
    if args.resolution < 2 or args.scale_steps < 2 or args.starts < 1:
        parser.error("resolution 和 scale-steps 至少为 2，starts 至少为 1")
    if args.output is None:
        name = f"{args.source.stem}_to_{args.target.stem}.txt"
        args.output = Path(__file__).resolve().parent / "result" / name
    for cloud in (args.source, args.target):
        if args.output.exists() and args.output.samefile(cloud):
            parser.error("输出文件不能覆盖输入点云")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix = register(read_xyz(args.source), read_xyz(args.target), args)
    np.savetxt(args.output, matrix, fmt="%.10f")
    print(f"矩阵已保存到 {args.output.resolve()}")
    print(args.output.read_text(), end="")
