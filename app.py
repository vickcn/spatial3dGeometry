import LOGger
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
from spatial3dGeometry import p0, projectInitial
import json
import os

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_surface', methods=['POST'])
def generate_surface():
    try:
        # 從請求中獲取參數
        data = request.json
        fixed_height = float(data.get('fixed_height', 0.0))
        num_edge_points = int(data.get('num_edge_points', 8))
        num_random_points = int(data.get('num_random_points', 5))
        random_centred_points = float(data.get('random_centred_points', 0.5))
        interpolated_z_method = data.get('interpolated_z_method', 'cubic')
        smoothness = float(data.get('smoothness', 0.5))

        result = {}
        # 使用 p0 函數生成曲面
        if not p0(
            ret = result,
            height_range=(0, fixed_height),
            grid_resolution=100,
            RandomCentredPointNumbers=num_random_points,
            scaleRatios=[random_centred_points, random_centred_points, random_centred_points],
            smoothness=smoothness,
            interpolated_z_method=interpolated_z_method
        ):
            raise Exception("生成曲面失敗")

        # 從 handler 的 exports 中獲取結果
        # result = handler.exports.get('result', {})
        # if not result:
        #     raise Exception("無法獲取曲面數據")

        # 將 numpy 數組轉換為列表
        Z_surface = LOGger.dcp(result['outputs']['Z_surface'])
        Z_surface = np.where(np.isnan(Z_surface), 0, Z_surface)

        # LOGger.addDebug(str(result['outputs']['Z_surface']), stamps=['Z'])
        X_surface = result['outputs']['X_surface'].tolist()
        Y_surface = result['outputs']['Y_surface'].tolist()
        Z_surface = Z_surface.tolist()

        # 获取内部点数据
        x_random = result['inputs']['x_random'].tolist()
        y_random = result['inputs']['y_random'].tolist()
        z_random = result['inputs']['z_random'].tolist()

        # 获取边缘点数据
        x_edge = result['inputs']['x_smooth'].tolist()
        y_edge = result['inputs']['y_smooth'].tolist()
        z_edge = result['inputs']['z_smooth'].tolist()

        return jsonify({
            'status': 'success',
            'data': {
                'X_surface': X_surface,
                'Y_surface': Y_surface,
                'Z_surface': Z_surface,
                'inner_points': {
                    'x': x_random,
                    'y': y_random,
                    'z': z_random
                },
                'edge_points': {
                    'x': x_edge,
                    'y': y_edge,
                    'z': z_edge
                }
            }
        })
    except Exception as e:
        LOGger.exception_process(e,logfile='',colora=LOGger.FAIL)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 
