# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:46:59 2021

@author: ian.ko
"""
import os
import struct
import visualization3 as vs3
import numpy as np
# import pyodbc as po
import pandas as pd
import platform as pf
from datetime import datetime as dt
import dataframeprocedure as DFP
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev, griddata
from scipy.ndimage import gaussian_filter
# 使用RBF插值并添加斜率约束
from scipy.interpolate import Rbf
from collections import defaultdict
from matplotlib.path import Path
LOGger = DFP.LOGger
dcp = LOGger.dcp
cv2 = vs3.cv2

# from qd.cae import dyna
# from qd.cae.dyna import D3plot
# import pypost

import TWSGCAE_preprocessing as TWSGCAE
#%%
if(False):
    __file__ = 'spatial3dGeometry.py'
curfn = os.path.basename(__file__)
theme = curfn.replace('.py','')
json_file = '%s_buffer.json'%theme
#%%
def generateRandomPartEdge(m=50, height_range=(-1, 1), smoothness=0.5, base_smoothness=0.5, oscillation=5, constant_height=None):
    """
    Generate a closed 3D curve with m points, shaped like a golf club head.

    Parameters:
        m (int): Number of points in the contour.
        height_range (tuple): Range of heights (z-values) for the points.
        smoothness (float): Smoothing factor for the 3D spline curve.
        base_smoothness (float): Smoothing factor for the base plane spline curve.
        oscillation (float): Maximum deviation of the base contour radius from its mean.
        constant_height (float): If provided, all points will have this z-value.

    Returns:
        tuple: Arrays of x, y, z coordinates of the smooth closed curve.
    """
    # 生成基本角度
    angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
    
    # 定義高爾夫球頭的基本形狀參數
    mean_radius = 7.5  # 平均半徑
    front_radius = mean_radius * 1.2  # 擊球面半徑較大
    back_radius = mean_radius * 0.8   # 背面半徑較小
    
    # 生成不對稱的輪廓
    radius = np.zeros_like(angles)
    for i, angle in enumerate(angles):
        # 擊球面（正面）
        if -np.pi/4 <= angle <= np.pi/4:
            radius[i] = front_radius + np.random.uniform(-oscillation*0.5, oscillation*0.5)
        # 背面
        elif np.pi*3/4 <= angle <= np.pi*5/4:
            radius[i] = back_radius + np.random.uniform(-oscillation*0.5, oscillation*0.5)
        # 側面
        else:
            radius[i] = mean_radius + np.random.uniform(-oscillation, oscillation)
    
    # 生成 x, y 坐標
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    # 確保曲線封閉
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    
    # 擬合基礎平面曲線
    base_tck, base_u = splprep([x, y], s=base_smoothness, per=True)
    base_u_fine = np.linspace(0, 1, 100)
    x_base_smooth, y_base_smooth = splev(base_u_fine, base_tck)
    
    # 生成 z 值（高度）
    if constant_height is not None:
        z = np.full(m + 1, constant_height)
    else:
        z = np.zeros(m + 1)
        # 頂部較高，底部較平
        for i, angle in enumerate(angles):
            # 頂部（擊球面）
            if -np.pi/4 <= angle <= np.pi/4:
                z[i] = height_range[1] * 0.8 + np.random.uniform(-0.1, 0.1)
            # 底部
            elif np.pi*3/4 <= angle <= np.pi*5/4:
                z[i] = height_range[0] + np.random.uniform(-0.05, 0.05)
            # 側面
            else:
                z[i] = np.mean(height_range) + np.random.uniform(-0.1, 0.1)
        z[-1] = z[0]  # 閉合點
    
    # 擬合 3D 曲線
    tck, u = splprep([x, y, z], s=smoothness, per=True)
    u_fine = np.linspace(0, 1, 100)
    x_smooth, y_smooth, z_smooth = splev(u_fine, tck)
    
    return x_smooth, y_smooth, z_smooth, x_base_smooth, y_base_smooth

def setRandomPartEdge(handler, **kwags):
    m = handler.EdgePointNumbers
    height_range = handler.height_range
    smoothness = handler.smoothness
    base_smoothness = handler.base_smoothness
    oscillation = handler.oscillation
    constant_height = handler.constant_height
    x_smooth, y_smooth, z_smooth, x_base_smooth, y_base_smooth = generateRandomPartEdge(
        m=m, height_range=height_range, smoothness=smoothness, 
        base_smoothness=base_smoothness, oscillation=oscillation, constant_height=constant_height)
    handler.x_smooth = x_smooth
    handler.y_smooth = y_smooth
    handler.z_smooth = z_smooth
    handler.x_base_smooth = x_base_smooth
    handler.y_base_smooth = y_base_smooth
    return True

def generateRandomInnerPoints(x_base, y_base, n=25, height_range=(-1, 1), z_mean=None, z_std=None, mstd=0.2, max_z_diff=None, **kwags):
    """
    Generate random points within a closed base region and assign random heights.

    Parameters:
        x_base (array): x-coordinates of the base closed curve.
        y_base (array): y-coordinates of the base closed curve.
        n (int): Number of random points to generate within the region.
        height_range (tuple): Range of heights (z-values) for the points.
        z_mean (float): Mean height of the 3D curve.
        z_std (float): Standard deviation of the heights of the 3D curve.
        max_z_diff (float): Maximum allowed difference between z values of points (relative to z_std).

    Returns:
        tuple: Arrays of x, y, z coordinates of the random points with heights.
    """

    # Create a path object for the base region
    base_path = Path(np.column_stack((x_base, y_base)))

    # Generate random points within the bounding box of the base region
    min_x, max_x = np.min(x_base), np.max(x_base)
    min_y, max_y = np.min(y_base), np.max(y_base)
    points = []
    while len(points) < n:
        random_points = np.random.uniform((min_x, min_y), (max_x, max_y), size=(n * 2, 2))
        inside = random_points[base_path.contains_points(random_points)]
        points.extend(inside[: n - len(points)])

    points = np.array(points)
    x_points, y_points = points[:, 0], points[:, 1]

    # Generate initial z values
    if z_mean is not None and z_std is not None:
        z_min = z_mean - mstd * z_std
        z_max = z_mean + mstd * z_std
        z_points = np.random.uniform(z_min, z_max, size=n)
    else:
        z_points = np.random.uniform(height_range[0], height_range[1], size=n)

    # If max_z_diff is specified, adjust z values to meet the constraint
    if max_z_diff is not None and n > 1:
        # Calculate the maximum allowed difference (relative to z_std or height range)
        if z_std is not None:
            max_diff = z_std * max_z_diff
        else:
            max_diff = (height_range[1] - height_range[0]) * max_z_diff

        # Sort points by z value
        sorted_indices = np.argsort(z_points)
        sorted_z = z_points[sorted_indices]

        # Adjust z values to ensure maximum difference is not exceeded
        for i in range(1, len(sorted_z)):
            if sorted_z[i] - sorted_z[i-1] > max_diff:
                sorted_z[i] = sorted_z[i-1] + max_diff

        # Assign adjusted z values back to original points
        z_points[sorted_indices] = sorted_z

    return x_points, y_points, z_points

def setRandomInnerPoints(handler, **kwags):
    n = handler.RandomInnerPointNumbers
    height_range = handler.height_range
    z_smooth = handler.z_smooth
    z_mean = np.mean(z_smooth)
    z_std = np.std(z_smooth)
    x_base_smooth = handler.x_base_smooth
    y_base_smooth = handler.y_base_smooth
    x_random, y_random, z_random = generateRandomInnerPoints(
        x_base_smooth, y_base_smooth, n=n, height_range=height_range, z_mean=z_mean, z_std=z_std)
    handler.x_random = x_random
    handler.y_random = y_random
    handler.z_random = z_random
    return True

def generateCenteredRandomPoints(x_base, y_base, z_base, n=5, scaleRatios=None, scaleRatioDefault=0.03, max_z_diff=0.2, **kwags):
    """
    Generate random points concentrated towards the center of a closed curve.

    Parameters:
        x_base (array): x-coordinates of the base closed curve.
        y_base (array): y-coordinates of the base closed curve.
        z_base (array): z-coordinates of the base closed curve.
        n (int): Number of random points to generate.
        max_z_diff (float): Maximum allowed difference between z values of points (relative to z range).
        scaleRatios (list): Scale ratios for x, y, z coordinates.
        scaleRatioDefault (float): Default scale ratio if not specified.

    Returns:
        tuple: Arrays of x, y, z coordinates for the random points.
    """
    # 創建封閉曲線的 Path 對象
    base_path = Path(np.column_stack((x_base, y_base)))

    center_x = np.mean(x_base)
    center_y = np.mean(y_base)
    center_z = np.mean(z_base)
    oscillationBase_z = (np.max(z_base) - np.min(z_base)) if(np.max(z_base) - np.min(z_base)>0.01) else 1

    scaleRatios = LOGger.mylist(scaleRatios if(isinstance(scaleRatios, list)) else [scaleRatioDefault])
    print('scaleRatios', scaleRatios, 'oscillationBase_z', oscillationBase_z)

    # 生成點直到獲得足夠的內部點
    ta = LOGger.dt.now()
    valid_points = np.empty((0, 2))  # 存儲所有有效點
    while valid_points.shape[0] < n:
        # 生成候選點（每次多生成一些點以提高效率）
        num_candidates = max((n - valid_points.shape[0]) * 2, n * 2)
        x_candidates = np.random.normal(loc=center_x, 
                                      scale=scaleRatios.get(0, scaleRatioDefault) * (np.max(x_base) - np.min(x_base)), 
                                      size=num_candidates)
        y_candidates = np.random.normal(loc=center_y, 
                                      scale=scaleRatios.get(1, scaleRatioDefault) * (np.max(y_base) - np.min(y_base)), 
                                      size=num_candidates)
        
        # 檢查哪些點在曲線內部
        points = np.column_stack((x_candidates, y_candidates))
        inside_mask = base_path.contains_points(points)
        new_valid_points = points[inside_mask]
        
        # 合併新的有效點
        valid_points = np.vstack([valid_points, new_valid_points])
        
        if valid_points.shape[0] < n:
            print('n:%d<(%d)'%(valid_points.shape[0], n))

        if((LOGger.dt.now() - ta).total_seconds() > 60):
            raise TimeoutError('生成中心點超時！')
    
    # 只取需要的點數
    valid_points = valid_points[:n]
    x_points = valid_points[:, 0]
    y_points = valid_points[:, 1]
    
    # 計算邊緣高度的範圍
    # z_min = np.min(z_base)
    z_range = np.max(z_base)
    
    # 生成初始 z 值，確保與邊緣高度的差異不超過 max_z_diff 倍
    z_points = np.zeros(n)
    
    # 生成第一個點的 z 值
    z_points[0] = np.random.uniform(
        center_z + 0.1 * max_z_diff * z_range,
        center_z + max_z_diff * z_range
    )
    
    # 生成其他點的 z 值
    for i in range(1, n):
        # 確保每個點的 z 值與邊緣高度的差異不超過 max_z_diff 倍
        z_points[i] = np.random.uniform(
            z_points[0] - 0.045*max_z_diff * z_range,
            z_points[0] + 0.045*max_z_diff * z_range
        )

    # 確保所有數組長度相同
    assert len(x_points) == len(y_points) == len(z_points) == n, \
           f"點數不匹配：x={len(x_points)}, y={len(y_points)}, z={len(z_points)}, n={n}"

    return x_points, y_points, z_points

def setRandomCentrePoints(handler, scaleRatios=None, scaleRatioDefault=0.03, **kwags):
    x_base_smooth = handler.x_base_smooth
    y_base_smooth = handler.y_base_smooth
    n = handler.RandomCentredPointNumbers
    scaleRatios = handler.scaleRatios
    z_smooth = handler.z_smooth
    x_random, y_random, z_random = generateCenteredRandomPoints(
        x_base_smooth, y_base_smooth, z_smooth, n=n, scaleRatios=scaleRatios, scaleRatioDefault=scaleRatioDefault)
    handler.x_random = x_random
    handler.y_random = y_random
    handler.z_random = z_random
    return True

def extendSurfaceFromCurve(x_curve, y_curve, z_curve, grid_resolution=50, smoothness=0.1, random_points=None,
                           interpolated_z_method='cubic', handler=None, max_slope=None):
    """
    Generate a smooth surface within a closed 3D curve with slope constraints.
    The surface will exactly pass through both the boundary curve and internal points.

    Parameters:
        x_curve (array): x-coordinates of the closed 3D curve.
        y_curve (array): y-coordinates of the closed 3D curve.
        z_curve (array): z-coordinates of the closed 3D curve.
        grid_resolution (int): Resolution of the grid for the surface.
        smoothness (float): Smoothing factor for interpolating the surface.
        random_points (tuple): Random points (x, y, z) to ensure they pass through the surface.
        max_slope (float): Maximum allowed slope between points. If None, no slope constraint.

    Returns:
        tuple: Meshgrid arrays (X, Y, Z) representing the smooth surface.
    """
    
    # Create a path object for the closed 2D projection of the curve
    base_path = Path(np.column_stack((x_curve, y_curve)))

    # Generate a grid within the bounding box of the curve
    min_x, max_x = np.min(x_curve), np.max(x_curve)
    min_y, max_y = np.min(y_curve), np.max(y_curve)
    grid_x, grid_y = np.meshgrid(
        np.linspace(min_x, max_x, grid_resolution),
        np.linspace(min_y, max_y, grid_resolution)
    )

    if(isinstance(getattr(handler, 'exports', None), dict)):  
        exports = handler.exports
        exports['extendParams'] = {'grid_resolution':grid_resolution, 'smoothness':smoothness, 'interpolated_z_method':interpolated_z_method}

    # Mask points outside the closed curve
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    inside_mask = base_path.contains_points(grid_points)
    inside_points = grid_points[inside_mask]

    # Combine curve and random points for interpolation
    interpolation_points = np.column_stack((x_curve, y_curve, z_curve))
    if random_points is not None:
        interpolation_points = np.vstack((interpolation_points, np.column_stack(random_points)))

    # 計算所有點對之間的最大斜率
    if max_slope is None:
        slopes = []
        for i in range(len(interpolation_points)):
            for j in range(i + 1, len(interpolation_points)):
                p1 = interpolation_points[i]
                p2 = interpolation_points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                dz = p2[2] - p1[2]
                dist_xy = np.sqrt(dx*dx + dy*dy)
                if dist_xy > 0:
                    slope = abs(dz / dist_xy)
                    slopes.append(slope)
        max_slope = np.mean(slopes) + np.std(slopes)

    # 使用精確的 RBF 插值，確保通過所有控制點
    rbf = Rbf(interpolation_points[:, 0], 
              interpolation_points[:, 1], 
              interpolation_points[:, 2], 
              function='linear',     # 使用線性基函數確保精確插值
              smooth=0.0)           # 設置為0確保精確通過控制點

    # 計算初始插值結果
    interpolated_z = rbf(inside_points[:, 0], inside_points[:, 1])

    # 創建 KDTree 用於快速查找最近點
    tree = KDTree(interpolation_points[:, :2])

    # 對每個網格點應用斜率約束
    for i in range(len(inside_points)):
        # 找到最近的4個控制點
        distances, indices = tree.query(inside_points[i], k=4)
        
        # 計算這些點的加權平均高度
        weights = 1.0 / (distances + 1e-10)
        target_heights = interpolation_points[indices, 2]
        weighted_height = np.average(target_heights, weights=weights)
        
        # 計算允許的最大高度差
        min_dist = np.min(distances)
        max_dz = max_slope * min_dist

        # 調整高度，但保持對控制點的精確插值
        current_height = interpolated_z[i]
        
        # 檢查當前點是否是控制點
        is_control_point = False
        for control_point in interpolation_points:
            if np.allclose(inside_points[i], control_point[:2], rtol=1e-10, atol=1e-10):
                is_control_point = True
                break
        
        # 如果不是控制點，才進行高度調整
        if not is_control_point:
            height_diff = weighted_height - current_height
            if abs(height_diff) > max_dz:
                # 使用平滑過渡
                sign = np.sign(height_diff)
                interpolated_z[i] = current_height + sign * max_dz

    # 創建最終的曲面網格
    z_surface = np.full_like(grid_x, np.nan)
    z_surface.ravel()[inside_mask] = interpolated_z

    # 應用局部平滑，但保持控制點不變
    valid_mask = ~np.isnan(z_surface)
    if np.any(valid_mask):
        # 創建控制點的掩碼
        control_points_mask = np.zeros_like(z_surface, dtype=bool)
        for point in interpolation_points:
            x, y = point[:2]
            i = np.abs(grid_x[0] - x).argmin()
            j = np.abs(grid_y[:, 0] - y).argmin()
            control_points_mask[j, i] = True
        
        # 只對非控制點進行平滑
        smooth_mask = valid_mask & ~control_points_mask
        if np.any(smooth_mask):
            # 保存控制點的原始值
            original_values = z_surface.copy()
            # 平滑整個表面
            z_surface[valid_mask] = gaussian_filter(z_surface[valid_mask], sigma=0.5)
            # 恢復控制點的值
            z_surface[control_points_mask] = original_values[control_points_mask]

    return grid_x, grid_y, z_surface

def saveToD3plot(X_surface, Y_surface, Z_surface, file):
    """
    Save the surface data to a binary file with a simple format for client-side import.

    Format:
    - Header (12 bytes):
        - grid_rows (int32): Number of rows in the grid
        - grid_cols (int32): Number of columns in the grid
        - version (int32): Format version (1)
    - Data (float32 arrays):
        - X_surface (flattened)
        - Y_surface (flattened)
        - Z_surface (flattened)

    Parameters:
        X_surface, Y_surface, Z_surface: Surface mesh grids
        file: Output file path
    """
    grid_rows, grid_cols = X_surface.shape
    version = 1

    with open(file, 'wb') as f:
        # Write header
        f.write(struct.pack('iii', grid_rows, grid_cols, version))
        
        # Write flattened surface data
        X_surface.astype(np.float32).ravel().tofile(f)
        Y_surface.astype(np.float32).ravel().tofile(f)
        Z_surface.astype(np.float32).ravel().tofile(f)

    return True

def readFromD3plot(file):
    """
    Read surface data from the binary file.

    Returns:
        tuple: (X_surface, Y_surface, Z_surface) as 2D arrays
    """
    with open(file, 'rb') as f:
        # Read header
        grid_rows, grid_cols, version = struct.unpack('iii', f.read(12))
        
        # Read data
        size = grid_rows * grid_cols
        X_surface = np.fromfile(f, dtype=np.float32, count=size).reshape((grid_rows, grid_cols))
        Y_surface = np.fromfile(f, dtype=np.float32, count=size).reshape((grid_rows, grid_cols))
        Z_surface = np.fromfile(f, dtype=np.float32, count=size).reshape((grid_rows, grid_cols))
        
    return X_surface, Y_surface, Z_surface

def solveSurfaceFromCurve(handler):
    x_smooth = handler.x_smooth
    y_smooth = handler.y_smooth
    z_smooth = handler.z_smooth
    grid_resolution = handler.grid_resolution
    smoothness = handler.smoothness
    x_random = handler.x_random
    y_random = handler.y_random
    z_random = handler.z_random
    handler.X_surface, handler.Y_surface, handler.Z_surface = extendSurfaceFromCurve(
        x_smooth, y_smooth, z_smooth, grid_resolution=grid_resolution, smoothness=smoothness, random_points=(x_random, y_random, z_random),
        interpolated_z_method=handler.interpolated_z_method, handler=handler)
    
    # 保存輸入和輸出數據
    exports = handler.exports
    exports['inputs'] = {}
    exports['inputs']['x_smooth'] = handler.x_smooth
    exports['inputs']['y_smooth'] = handler.y_smooth
    exports['inputs']['z_smooth'] = handler.z_smooth
    exports['inputs']['x_random'] = handler.x_random
    exports['inputs']['y_random'] = handler.y_random
    exports['inputs']['z_random'] = handler.z_random
    
    exports['outputs'] = {}
    exports['outputs']['X_surface'] = handler.X_surface
    exports['outputs']['Y_surface'] = handler.Y_surface
    exports['outputs']['Z_surface'] = handler.Z_surface
    
    # 保存為二進制文件
    # surface_file = os.path.join(handler.exp_fd, 'surface.d3p')
    # saveToD3plot(handler.X_surface, handler.Y_surface, handler.Z_surface, surface_file)
    
    return True

def drawRandomPartEdge(ax, x_smooth, y_smooth, z_smooth, x_base_smooth, y_base_smooth):
    ax.plot(x_smooth, y_smooth, z_smooth, label='Smooth Closed Curve')
    ax.scatter(x_smooth, y_smooth, z_smooth, c=z_smooth, cmap='viridis', s=10)
    ax.plot(x_base_smooth, y_base_smooth, np.zeros_like(x_base_smooth), label='Base Projection', linestyle='--', color='gray')
    # Plot vertical projection lines
    for x, y, z in zip(x_smooth, y_smooth, z_smooth):
        ax.plot([x, x], [y, y], [z, 0], color='black', alpha=0.3, linestyle=':')
    
def drawInnerPoints(ax, x_random, y_random, z_random):
    # Plot the random points with heights
    ax.scatter(x_random, y_random, z_random, c=z_random, cmap='coolwarm', s=20, label='Random Points')
    
    # Plot vertical projection lines for the random points
    for x, y, z in zip(x_random, y_random, z_random):
        ax.plot([x, x], [y, y], [z, 0], color='purple', alpha=0.3, linestyle=':')
        ax.text(x,y,z,'(%s)'%DFP.parse((x,y,z)))
    ax.scatter(x_random, y_random, np.zeros_like(x_random), cmap='coolwarm', s=20, label='Random Points', marker='x', c='black')
    # ax.plot(x_base_smooth, y_base_smooth, np.zeros_like(x_base_smooth), label='Base Projection', linestyle='--', color='gray')
    # Plot vertical projection lines
    # for x, y, z in zip(x_smooth, y_smooth, z_smooth):
    #     ax.plot([x, x], [y, y], [z, 0], color='black', alpha=0.3, linestyle=':')
    
def drawRandomPartEdgeAndRandomInnerPoints(ax, x_smooth, y_smooth, z_smooth, x_base_smooth, y_base_smooth, x_random, y_random, z_random):
    drawRandomPartEdge(ax, x_smooth, y_smooth, z_smooth, x_base_smooth, y_base_smooth)
    drawInnerPoints(ax, x_random, y_random, z_random)

def drawAxisFormat(ax):
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

def plotSituationAndSolved(handler):
    fig = vs3.plt.Figure(figsize=handler.figsize)
    ax = fig.add_subplot(111, projection='3d')
    drawRandomPartEdgeAndRandomInnerPoints(ax, handler.x_smooth, handler.y_smooth, handler.z_smooth, 
                                           handler.x_base_smooth, handler.y_base_smooth, 
                                           handler.x_random, handler.y_random, handler.z_random)
    
    drawAxisFormat(ax)
    file = os.path.join(handler.exp_fd, 'situation.jpg')
    LOGger.CreateFile(file, lambda f:vs3.end(fig, file=f))
    # Plot the generated surface
    # LOGger.addDebug(str(handler.Z_surface), stamps=['surface'])
    ax.plot_surface(handler.X_surface, handler.Y_surface, handler.Z_surface, cmap='viridis', alpha=0.3)
    drawAxisFormat(ax)
    
    # 使用 plt.show() 显示
    # vs3.plt.show()
    file = os.path.join(handler.exp_fd, 'solved.jpg')
    LOGger.CreateFile(file, lambda f:vs3.end(fig, file=f))
    if(not exporting(handler)):
        return False
    return True

def saveRandomPartEdge():
    # Generate and plot the curve
    x_smooth, y_smooth, z_smooth, x_base_smooth, y_base_smooth = generateRandomPartEdge(
        m=50, height_range=(0.2, 0), smoothness=0.95, base_smoothness=0.95, oscillation=0.1)
    
    fig = vs3.plt.Figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    drawRandomPartEdge(ax, x_smooth, y_smooth, z_smooth, x_base_smooth, y_base_smooth)
    drawAxisFormat(ax)
    file = 'saveRandomPartEdge.jpg'
    LOGger.CreateFile(file, lambda f:vs3.end(fig, file=f))

def saveRandomPartEdgeAndRandomInnerPoints():
    height_range = (0.2, 0)
    # Generate and plot the curve
    x_smooth, y_smooth, z_smooth, x_base_smooth, y_base_smooth = generateRandomPartEdge(
        m=50, height_range=height_range, smoothness=0.95, base_smoothness=0.95, oscillation=0.1)
    z_mean = np.mean(z_smooth)
    z_std = np.std(z_smooth)
    x_random, y_random, z_random = generateRandomInnerPoints(x_base_smooth, y_base_smooth, n=25, height_range=height_range, z_mean=z_mean, z_std=z_std)
    
    fig = vs3.plt.Figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    drawRandomPartEdgeAndRandomInnerPoints(ax, x_smooth, y_smooth, z_smooth, x_base_smooth, y_base_smooth, x_random, y_random, z_random)
    drawAxisFormat(ax)
    file = 'saveRandomPartEdgeAndRandomInnerPoints.jpg'
    LOGger.CreateFile(file, lambda f:vs3.end(fig, file=f))
    
def saveGeneratingInnerSurfaceFromRandomPartEdgeAndRandomInnerPoints(handler=None, **kwags):
    if(not setRandomPartEdge(handler)):
        return False
    if(not setRandomInnerPoints(handler)):
        return False
    if(not solveSurfaceFromCurve(handler)):
        return False
    if(not plotSituationAndSolved(handler)):
        return False
    return True
    

def saveGeneratingInnerSurfaceFromRandomPartEdgeAndRandomCentredPoints(handler=None, **kwags):
    if(not setRandomPartEdge(handler)):
        return False
    if(not setRandomCentrePoints(handler)):
        return False
    if(not solveSurfaceFromCurve(handler)):
        return False
    if(not plotSituationAndSolved(handler)):
        return False
    return True
    
def exporting(handler, exp_fd=None):
    exports = handler.exports
    exp_fd = exp_fd if(LOGger.isinstance_not_empty(exp_fd, str)) else getattr(handler, 'exp_fd', '.')
    DFP.joblib.dump(exports, os.path.join(exp_fd, 'exports.pkl'))
    return True

#%%
def projectInitial(exp_fd_default='.', figsize=(20,20), smoothness=0.95, base_smoothness=0.95, oscillation=0.1,
                   EdgePointNumbers=50, constant_height=0.2,
                   interpolated_z_method='cubic', **kwags):
    handler = LOGger.mystr()
    project_buffer = kwags.get('project_buffer', {})
    if(LOGger.isinstance_not_empty(project_buffer.get('exp_fd'), str)):
        handler.exp_fd = dcp(project_buffer['exp_fd'])
    else:
        handler.exp_fd = exp_fd_default
        project_buffer.update({'exp_fd': handler.exp_fd})
    if(os.path.exists(handler.exp_fd) and handler.exp_fd!='.'):
        LOGger.removefile(handler.exp_fd)
    if(not os.path.exists(handler.exp_fd)):
        LOGger.CreateContainer(handler.exp_fd)
    handler.logfile = os.path.join(handler.exp_fd, 'log.txt')
    handler.addlog = LOGger.addloger(logfile=handler.logfile)
    handler.stamps = kwags.get('stamps', [])
    handler.figsize = figsize
    handler.smoothness = smoothness
    handler.base_smoothness = base_smoothness
    handler.oscillation = oscillation
    handler.interpolated_z_method = interpolated_z_method
    handler.constant_height = constant_height
    handler.EdgePointNumbers = EdgePointNumbers
    handler.exports = {}
    # handler.config_file = config_file if(LOGger.isinstance_not_empty(config_file, str)) else 'config.json'
    # handler.config = LOGger.load_json(handler.config_file)
    
    # handler.model_fn = 'model'
    return handler

def p0(excludingHeaderFile=None, default_exp_fd='%s_p0'%(os.path.basename(__file__).split('.')[0]), height_range = (0,0.2), 
       grid_resolution=100, scaleRatios=None, RandomCentredPointNumbers=5, handler=None, ret=None, **kwags):
    handler = projectInitial(exp_fd_default=default_exp_fd, **kwags)
    handler.projectName = 'p0'
    handler.height_range = height_range #generate需要
    handler.grid_resolution = grid_resolution #solve需要
    LOGger.addDebug(getattr(handler, 'RandomCentredPointNumbers', '?'))
    handler.RandomCentredPointNumbers = RandomCentredPointNumbers
    handler.scaleRatios = scaleRatios if(isinstance(scaleRatios, list)) else [0.03,0.03,0.1] #randomCentre需要
    project_buffer = kwags.get('project_buffer', {})
    project_buffer['exp_fd'] = handler.exp_fd
    try:
        LOGger.addDebug('a')
        if(not saveGeneratingInnerSurfaceFromRandomPartEdgeAndRandomCentredPoints(handler=handler, **kwags)):
            return False
        LOGger.addDebug('b')
        if(isinstance(ret, dict)):
            ret.update(handler.exports)
    except Exception as e:
        LOGger.exception_process(e, logfile='')
        LOGger.addDebug('c')
        return False
    return True

def method_activation(stg):
    try:
        method = eval(stg)
        return method
    except:
        print('method invalid:%s!!!!'%stg)
        return None

def scenario():
    args_history = LOGger.load_json(json_file) if(os.path.exists(json_file)) else {}
    parser = LOGger.myArgParser()
    # parser.add_argument("-i", "--source_fd", type=str, help="輸入資料夾")
    parser.add_argument("-prmth", "--project_method_stg", type=str, help="指定蒐集方式(`p0`)\
                        (p0:saveGeneratingInnerSurfaceFromRandomPartEdgeAndRandomCentredPoints; \
                         ", default='p0')
    parser.add_argument("-sm", "--smoothness", type=float, help="平滑性(`0.95`)", default=args_history.get('smoothness', 0.95))
    parser.add_argument("-bsm", "--base_smoothness", type=float, help="邊緣封閉曲線的平滑性(`0.95`)", default=args_history.get('base_smoothness', 0.95))
    parser.add_argument("-oc", "--oscillation", type=float, help="邊緣封閉曲線的擾動程度(`0.1`)", default=args_history.get('oscillation', 0.1))
    parser.add_argument("-ht", "--constant_height", type=float, help="邊緣固定高度(`0.2`/`None`)", default=args_history.get('constant_height', 0.2))
    parser.add_argument("-m", "--EdgePointNumbers", type=int, help="邊緣的駐點數量", default=args_history.get('EdgePointNumbers', 50))
    parser.add_argument("-n", "--RandomCentredPointNumbers", type=int, help="給定中心叢聚駐點的數量", 
                        default=args_history.get('RandomCentredPointNumbers', 5))
    parser.add_argument("-srs", "--scaleRatios", type=eval, help="給定中心叢聚點的離散性`[0.03,0.03,0.1]`", 
                        default=args_history.get('scaleRatios', [0.03,0.03,0.1]))
    parser.add_argument("-hr", "--height_range", type=eval, help="給定隨機空間點的高度限制`(0,0.2)`", 
                        default=args_history.get('height_range', (0,0.2)))
    parser.add_argument("-zm", "--interpolated_z_method", type=str, help="解空間曲面的方式(cubic)", 
                        default=args_history.get('interpolated_z_method', 'cubic'))
    parser.add_argument("-g", "--grid_resolution", type=int, help="解曲面的網格點數量(100)", 
                        default=args_history.get('interpolated_z_method', 100))
    parser.add_argument("-mzd", "--max_z_diff", type=float, help="內部點z值的最大差異限制(0.2)", 
                        default=args_history.get('max_z_diff', 0.2))
    parser.add_argument("-o", "--exp_fd", type=str, help="暫存輸出資料夾", default=None)
    args = parser.parse_args()
    project_buffer = vars(args)
    print('project_buffer:', LOGger.stamp_process('',project_buffer,':','[',']','\n'))
    # if(not LOGger.isinstance_not_empty(project_buffer['source_fd'], str)):
    #     print('source `%s` error!!!'%project_buffer['source_fd'])
    #     return
    # if(not os.path.isdir(project_buffer['source_fd'])):
    #     print('source `%s` error!!!'%project_buffer['source_fd'])
    #     return
    # if(DFP.isiterable(project_buffer.get('deleteColors', None))):
    #     project_buffer['deleteColors'] = [x for x in project_buffer['deleteColors'] if DFP.isiterable(x)]
    report = {}
    project_method = method_activation(project_buffer.get('project_method_stg'))
    if(not project_method(project_buffer=project_buffer, report=report, **project_buffer)):
        return
    exp_fd = project_buffer.pop('exp_fd', '.')
    for k,v in project_buffer.items():
        LOGger.addlog(str(type(v)), logfile=os.path.join(exp_fd, 'log.txt'), stamps=[k])
    LOGger.CreateFile(os.path.join(exp_fd, os.path.basename(json_file)), lambda f:LOGger.save_json(project_buffer, f))
    if(report):
        print(report)
        DFP.project_record_ending(report, sheet_name='main', exp_fd=exp_fd, theme=theme)
    

#%%    
if(__name__=='__main__' ):
    scenario()
    


