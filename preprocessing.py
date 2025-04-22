# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:46:59 2021

@author: ian.ko
"""
import os
import visualization3 as vs3
import numpy as np
# import pyodbc as po
import pandas as pd
import platform as pf
from datetime import datetime as dt
import dataframeprocedure as DFP
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import KDTree
from collections import defaultdict
LOGger = DFP.LOGger
dcp = LOGger.dcp
cv2 = vs3.cv2

from qd.cae import dyna
# import pypost
#%%
if(False):
    __file__ = 'TWSGCAE_preprocessing.py'
curfn = os.path.basename(__file__)
theme = curfn.replace('.py','')
json_file = '%s_buffer.json'%theme
#%%
def loadSimulationData(file):
    d3plot_file = '..\\TWSGCAE\\STR\\d3plot'
    d3plot = dyna.D3plot(d3plot_file)
    node = d3plot.get_nodeByID(1)
    nodeCoord = node.get_coords()
    show_part_ids(d3plot)
    part = d3plot.get_partByID(2)
    elemIDs = part.get_element_ids()
    show_element_ids(part)
    
def searchSurface(nodes, faces):
    # 1. 建立 KD-Tree
    all_points = faces.reshape(-1, 3)  # 所有頂點展平
    tree = KDTree(all_points)
    
    # 2. 計算每個面的質心
    centroids = np.mean(faces, axis=1)  # 每個面的質心
    
    # 3. 查詢質心的鄰居
    face_counts = defaultdict(int)
    for centroid in centroids:
        # 查詢最近的頂點
        _, idx = tree.query(centroid, k=1)  # 最近的頂點索引
        face_counts[tuple(centroid)] += 1
    
    # 4. 提取只出現一次的面
    surface_faces = [face for face, centroid in zip(faces, centroids) if face_counts[tuple(centroid)] == 1]
    return surface_faces

def drawPart(part, ax, alpha=0.2):
    nodes = part.get_nodes()
    nodeCoords = np.array([node.get_coords() for node in nodes]).squeeze(axis=1)
    elements = part.get_elements()
    faces = np.array([[node.get_coords() for node in element.get_nodes()] for element in elements]).squeeze(axis=2) 
    poly3d = Poly3DCollection(faces, alpha=alpha, edgecolor='k')
    poly3d.set_facecolor((0.5, 0.8, 0.9, 0.7))  # 面顏色
    
    ax.add_collection3d(poly3d)
    # 繪製節點
    ax.scatter(nodeCoords[:, 0], nodeCoords[:, 1], nodeCoords[:, 2], color='r', s=2, label='Nodes')
    
    # 節點索引
    # for i, (x, y, z) in enumerate(nodeCoords):
    #     ax.text(x, y, z, f'{i}', color='blue')
    
    # 設置範圍
    ax.set_xlim([nodeCoords[:, 0].min(), nodeCoords[:, 0].max()])
    ax.set_ylim([nodeCoords[:, 1].min(), nodeCoords[:, 1].max()])
    ax.set_zlim([nodeCoords[:, 2].min(), nodeCoords[:, 2].max()])
    return True

def drawPartSurface(part, ax, alpha=0.2):
    nodes = part.get_nodes()
    nodeCoords = np.array([node.get_coords() for node in nodes]).squeeze(axis=1)
    elements = part.get_elements()
    faces = np.array([[node.get_coords() for node in element.get_nodes()] for element in elements]).squeeze(axis=2) 
    surface_faces_coords = searchSurface(nodeCoords, faces)
    
    poly3d = Poly3DCollection(surface_faces_coords, alpha=alpha, edgecolor='k')
    poly3d.set_facecolor((0.5, 0.8, 0.9, 0.7))  # 面顏色
    
    ax.add_collection3d(poly3d)
    # 繪製節點
    ax.scatter(nodeCoords[:, 0], nodeCoords[:, 1], nodeCoords[:, 2], color='r', s=2, label='Nodes')
    
    # 設置範圍
    ax.set_xlim([nodeCoords[:, 0].min(), nodeCoords[:, 0].max()])
    ax.set_ylim([nodeCoords[:, 1].min(), nodeCoords[:, 1].max()])
    ax.set_zlim([nodeCoords[:, 2].min(), nodeCoords[:, 2].max()])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return True

def visualizePart(part):
    fig = vs3.plt.Figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    if(not drawPart(part, ax)):
        return False
    fig.tight_layout()
    file = 'visualizePart.jpg'
    LOGger.CreateFile(file, lambda f:vs3.end(fig, file=f))
    return True
    
def visualizePartSurface(part):
    fig = vs3.plt.Figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    if(not drawPartSurface(part, ax)):
        return False
    fig.tight_layout()
    file = 'visualizePartSurface.jpg'
    LOGger.CreateFile(file, lambda f:vs3.end(fig, file=f))
    return True    
    
    
def show_element_ids(something):
    element_ids = something.get_element_ids()
    print("Element IDs:", element_ids)    

def show_part_ids(d3plot):
    part_ids = [x.get_id() for x in d3plot.get_parts()]
    print("Part IDs:", part_ids)  

def axisStresses2VonMises(frame_stress):
    von_mises_stress = np.sqrt(
        (frame_stress[:, 0] - frame_stress[:, 1])**2 +
        (frame_stress[:, 1] - frame_stress[:, 2])**2 +
        (frame_stress[:, 2] - frame_stress[:, 0])**2 +
        6 * (frame_stress[:, 3]**2 + frame_stress[:, 4]**2 + frame_stress[:, 5]**2)
    ) / np.sqrt(2)
    return von_mises_stress

def findMaxStress(d3plot):
    # 提取單元應力和對應的單元 ID
    solid_stresses = d3plot.get_solid_stress()  # (frames, elements, 6)
    element_ids = d3plot.get_element_ids()
    
    # 初始化最大值
    max_von_mises = -float("inf")
    max_frame = -1
    max_element_id = -1
    
    # 遍歷每個 frame，計算 Von Mises 應力
    for frame_idx, frame_stress in enumerate(solid_stresses):
        # 計算 Von Mises 應力
        von_mises_stresses = np.sqrt(
            0.5 * (
                (frame_stress[:, 0] - frame_stress[:, 1]) ** 2 + 
                (frame_stress[:, 1] - frame_stress[:, 2]) ** 2 + 
                (frame_stress[:, 2] - frame_stress[:, 0]) ** 2
            ) + 
            3 * (frame_stress[:, 3] ** 2 + frame_stress[:, 4] ** 2 + frame_stress[:, 5] ** 2)
        )
    
        # 找到該 frame 的最大 Von Mises 應力
        frame_max_von_mises = von_mises_stresses.max()
        element_index = von_mises_stresses.argmax()
    
        # 更新最大值
        if frame_max_von_mises > max_von_mises:
            max_von_mises = frame_max_von_mises
            max_frame = frame_idx
            max_element_id = element_ids[element_index]
    
    # 輸出結果
    print(f"最大 Von Mises 應力: {max_von_mises}")
    print(f"發生在 frame: {max_frame}")
    print(f"對應的 element ID: {max_element_id}")
    

def TWSGCAE_preprocessing(source_file, ):
    return True

#%%
def projectInitial(exp_fd_default='.', config_file='config.json', **kwags):
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
    handler.config_file = config_file if(LOGger.isinstance_not_empty(config_file, str)) else 'config.json'
    handler.config = LOGger.load_json(handler.config_file)
    handler.label_encoders = {}
    handler.model_fn = 'model'
    return handler

def p0(excludingHeaderFile=None, default_exp_fd='%s_p0'%(os.path.basename(__file__).split('.')[0]), figsize=(13,13), **kwags):
    """
    test if mdc activatable

    Parameters
    ----------
    config_file : TYPE
        DESCRIPTION.
    default_exp_fd : TYPE, optional
        DESCRIPTION. The default is '%s_p0'%(os.path.basename(__file__).split('.')[0]).
    **kwags : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    handler = projectInitial(exp_fd_default=default_exp_fd, **kwags)
    handler.projectName = 'p0'
    handler.excludingHeader = LOGger.load_json(excludingHeaderFile) if(LOGger.isinstance_not_empty(excludingHeaderFile, str)) else []
    handler.figsize = figsize
    project_buffer = kwags.get('project_buffer', {})
    project_buffer['exp_fd'] = handler.exp_fd
    try:
        if(not TWSGCAE_preprocessing(handler=handler, **kwags)):
            return False
    except Exception as e:
        LOGger.exception_process(e, logfile='')
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
    parser.add_argument("-i", "--source_fd", type=str, help="輸入資料夾")
    parser.add_argument("-th", "--threshold_bd", type=float, help="劃出輪廓的閥值", default=args_history.get('threshold_bd', 200))
    parser.add_argument("-mth", "--method", type=eval, help="劃出輪廓的方式", default=args_history.get('method', cv2.CHAIN_APPROX_SIMPLE))
    parser.add_argument("-eps", "--eps", type=float, help="顏色分類的DB半徑", default=args_history.get('eps', 0.14))
    parser.add_argument("-msp", "--min_sample", type=int, help="顏色分類的DB半徑min_samples", default=args_history.get('min_samples',10))
    parser.add_argument("-mc", "--maskColor", type=lambda x:(eval(x) if(x.lower()!='none') else None), help="遮罩顏色，盡量與要取出的顏色不同", 
                        default=args_history.get('maskColor', None))
    parser.add_argument("-dcs", "--deleteColors", type=lambda x:DFP.astype(x,d_type=eval,default=None), 
                        help="要刪除的顏色", default=args_history.get('deleteColors', None), nargs='*')
    parser.add_argument("-o", "--exp_fd", type=str, help="暫存輸出資料夾", default='test')
    args = parser.parse_args()
    project_buffer = vars(args)
    print('project_buffer:', LOGger.stamp_process('',project_buffer,':','[',']','\n'))
    if(not LOGger.isinstance_not_empty(project_buffer['source_fd'], str)):
        print('source `%s` error!!!'%project_buffer['source_fd'])
        return
    if(not os.path.isdir(project_buffer['source_fd'])):
        print('source `%s` error!!!'%project_buffer['source_fd'])
        return
    if(DFP.isiterable(project_buffer.get('deleteColors', None))):
        project_buffer['deleteColors'] = [x for x in project_buffer['deleteColors'] if DFP.isiterable(x)]
    report = {}
    if(not TWSGCAE_preprocessing(project_buffer=project_buffer, report=report, **project_buffer)):
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
    


