document.addEventListener('DOMContentLoaded', function() {
    const generateBtn = document.getElementById('generate-btn');
    const plotContainer = document.getElementById('plot-container');
    let surfacePlot = null;  // 添加全局變量來存儲圖表實例

    // 创建加载动画元素
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-spinner';
    loadingDiv.innerHTML = '生成中...';
    loadingDiv.style.display = 'none';
    plotContainer.parentNode.insertBefore(loadingDiv, plotContainer.nextSibling);

    // 初始化空白图表
    const emptyData = [{
        type: 'surface',
        x: [[0]],
        y: [[0]],
        z: [[0]],
        colorscale: 'Viridis',
        opacity: 0.8
    }];
    const initialLayout = {
        title: '等待生成曲面...',
        autosize: true,
        margin: { l: 0, r: 0, b: 0, t: 30 },
        scene: {
            xaxis: { title: 'X軸' },
            yaxis: { title: 'Y軸' },
            zaxis: { title: 'Z軸' }
        },
        showlegend: true
    };
    Plotly.newPlot(plotContainer, emptyData, initialLayout).then(function(plot) {
        surfacePlot = plot;
    });

    generateBtn.addEventListener('click', async function() {
        try {
            // 显示加载动画
            loadingDiv.style.display = 'block';
            generateBtn.disabled = true;
            generateBtn.textContent = '生成中...';

            // 清空现有图表
            await Plotly.purge(plotContainer);

            // 獲取所有輸入值
            const params = {
                fixed_height: parseFloat(document.getElementById('fixed_height').value),
                num_edge_points: parseInt(document.getElementById('num_edge_points').value),
                num_random_points: parseInt(document.getElementById('num_random_points').value),
                random_centred_points: parseFloat(document.getElementById('random_centred_points').value),
                interpolated_z_method: document.getElementById('interpolated_z_method').value,
                smoothness: parseFloat(document.getElementById('smoothness').value)
            };

            // 發送請求到後端
            const response = await fetch('/generate_surface', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });

            const data = await response.json();

            if (data.status === 'error') {
                throw new Error(data.message);
            }

            // 准备数据
            const surfaceData = [
                // 曲面
                {
                    type: 'surface',
                    x: data.data.X_surface,
                    y: data.data.Y_surface,
                    z: data.data.Z_surface,
                    colorscale: 'Viridis',
                    opacity: 0.8,
                    showscale: true
                },
                // 內部點
                {
                    type: 'scatter3d',
                    x: data.data.inner_points.x,
                    y: data.data.inner_points.y,
                    z: data.data.inner_points.z,
                    mode: 'markers',
                    marker: {
                        size: 5,
                        color: 'red',
                        symbol: 'circle'
                    },
                    name: '內部點'
                },
                // 邊緣點
                {
                    type: 'scatter3d',
                    x: data.data.edge_points.x,
                    y: data.data.edge_points.y,
                    z: data.data.edge_points.z,
                    mode: 'lines+markers',
                    line: {
                        color: 'blue',
                        width: 3
                    },
                    marker: {
                        size: 5,
                        color: 'blue',
                        symbol: 'circle'
                    },
                    name: '邊緣點'
                }
            ];

            const layout = {
                title: '生成的曲面、內部點與邊緣點',
                autosize: true,
                margin: { l: 0, r: 0, b: 0, t: 30 },
                scene: {
                    camera: {
                        eye: {x: 1.5, y: 1.5, z: 1.5}
                    },
                    xaxis: { title: 'X軸' },
                    yaxis: { title: 'Y軸' },
                    zaxis: { title: 'Z軸' }
                },
                showlegend: true
            };

            // 更新圖表時保存實例
            await Plotly.newPlot(plotContainer, [], layout).then(function(plot) {
                surfacePlot = plot;
            });

            // 逐个添加数据
            for(let i = 0; i < surfaceData.length; i++) {
                await Plotly.addTraces(plotContainer, surfaceData[i]);
                await new Promise(resolve => setTimeout(resolve, 300)); // 每个元素之间的延迟
            }

        } catch (error) {
            alert('生成曲面時出錯：' + error.message);
        } finally {
            // 隐藏加载动画
            loadingDiv.style.display = 'none';
            generateBtn.disabled = false;
            generateBtn.textContent = '生成曲面';
        }
    });

    // 添加匯出按鈕的樣式
    const exportButtonStyle = `
        position: absolute;
        top: 10px;
        right: 10px;
        padding: 8px 16px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        z-index: 1000;
    `;

    // 創建匯出按鈕
    function createExportButton() {
        const exportButton = document.createElement('button');
        exportButton.textContent = '匯出曲面';
        exportButton.style.cssText = exportButtonStyle;
        
        // 添加懸停效果
        exportButton.addEventListener('mouseenter', () => {
            exportButton.style.backgroundColor = '#45a049';
            exportButton.style.transform = 'translateY(-1px)';
            exportButton.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
        });
        exportButton.addEventListener('mouseleave', () => {
            exportButton.style.backgroundColor = '#4CAF50';
            exportButton.style.transform = 'translateY(0)';
            exportButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        });
        
        // 添加點擊事件
        exportButton.addEventListener('click', exportSurfaceData);
        
        // 將按鈕添加到 plotContainer 中
        plotContainer.style.position = 'relative';  // 確保容器可以定位絕對定位的子元素
        plotContainer.appendChild(exportButton);
    }

    // 匯出曲面數據的函數
    async function exportSurfaceData() {
        try {
            if (!surfacePlot || !surfacePlot.data || !surfacePlot.data[0]) {
                showNotification('請先生成曲面再匯出！', 'error');
                return;
            }

            // 獲取當前的曲面數據
            const surfaceData = {
                X_surface: surfacePlot.data[0].x,
                Y_surface: surfacePlot.data[0].y,
                Z_surface: surfacePlot.data[0].z
            };

            // 生成 LS-DYNA keyword 格式的文件內容
            let keywordContent = '';
            
            // 添加標題
            keywordContent += '$# LS-DYNA Keyword file created by Surface Generator\n';
            keywordContent += '*KEYWORD\n';
            
            // 添加節點
            keywordContent += '*NODE\n';
            const nodes = [];
            for (let i = 0; i < surfaceData.X_surface.length; i++) {
                for (let j = 0; j < surfaceData.X_surface[i].length; j++) {
                    const nodeId = i * surfaceData.X_surface[i].length + j + 1;
                    const x = surfaceData.X_surface[i][j];
                    const y = surfaceData.Y_surface[i][j];
                    const z = surfaceData.Z_surface[i][j];
                    // 格式化節點數據：nodeId, x, y, z (LS-DYNA格式)
                    keywordContent += `${nodeId.toString().padStart(8)},${x.toFixed(3).padStart(16)},${y.toFixed(3).padStart(16)},${z.toFixed(3).padStart(16)}\n`;
                    nodes.push(nodeId);
                }
            }

            // 添加單元（使用四節點殼單元）
            keywordContent += '*ELEMENT_SHELL\n';
            let elementId = 1;
            const rows = surfaceData.X_surface.length;
            const cols = surfaceData.X_surface[0].length;
            
            for (let i = 0; i < rows - 1; i++) {
                for (let j = 0; j < cols - 1; j++) {
                    const n1 = i * cols + j + 1;
                    const n2 = i * cols + (j + 1) + 1;
                    const n3 = (i + 1) * cols + (j + 1) + 1;
                    const n4 = (i + 1) * cols + j + 1;
                    // 格式化單元數據：elementId, partId, n1, n2, n3, n4 (LS-DYNA格式)
                    keywordContent += `${elementId.toString().padStart(8)},${(1).toString().padStart(8)},${n1.toString().padStart(8)},${n2.toString().padStart(8)},${n3.toString().padStart(8)},${n4.toString().padStart(8)}\n`;
                    elementId++;
                }
            }

            // 添加部件定義
            keywordContent += '*PART\n';
            keywordContent += '$#                                                                         title\n';
            keywordContent += 'Surface Part\n';
            keywordContent += '$#     pid     secid       mid     eosid      hgid      grav    adpopt      tmid\n';
            keywordContent += '         1         1         1         0         0         0         0         0\n';

            // 添加截面屬性
            keywordContent += '*SECTION_SHELL\n';
            keywordContent += '$#   secid    elform      shrf       nip     propt   qr/irid     icomp     setyp\n';
            keywordContent += '         1         2  0.833333         2         1         0         0         1\n';
            keywordContent += '$#      t1        t2        t3        t4      nloc     marea      idof    edgset\n';
            keywordContent += '     1.000     1.000     1.000     1.000         0         0         0         0\n';

            // 添加材料屬性（使用彈性材料）
            keywordContent += '*MAT_ELASTIC\n';
            keywordContent += '$#     mid        ro         e        pr        da        db  not used\n';
            keywordContent += '         1 7.800E-09 2.100E+05  0.300000     0.000     0.000         0\n';

            // 結束關鍵字
            keywordContent += '*END\n';

            // 創建 Blob 對象
            const blob = new Blob([keywordContent], { type: 'text/plain' });

            // 創建下載鏈接
            const downloadLink = document.createElement('a');
            downloadLink.href = URL.createObjectURL(blob);
            downloadLink.download = 'surface.k';

            // 觸發下載
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);

            // 清理 URL 對象
            URL.revokeObjectURL(downloadLink.href);

            // 顯示成功消息
            showNotification('曲面已成功匯出為 LS-DYNA 格式！', 'success');
        } catch (error) {
            console.error('匯出失敗：', error);
            showNotification('匯出失敗，請稍後再試。', 'error');
        }
    }

    // 顯示通知的函數
    function showNotification(message, type = 'success') {
        const notification = document.createElement('div');
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 15px 25px;
            border-radius: 5px;
            color: white;
            font-size: 16px;
            z-index: 1000;
            animation: fadeInOut 3s ease-in-out;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            background-color: ${type === 'success' ? '#4CAF50' : '#f44336'};
        `;

        // 添加動畫樣式
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fadeInOut {
                0% { opacity: 0; transform: translate(-50%, -20px); }
                15% { opacity: 1; transform: translate(-50%, 0); }
                85% { opacity: 1; transform: translate(-50%, 0); }
                100% { opacity: 0; transform: translate(-50%, -20px); }
            }
        `;
        document.head.appendChild(style);

        document.body.appendChild(notification);
        setTimeout(() => {
            document.body.removeChild(notification);
            document.head.removeChild(style);
        }, 3000);
    }

    // 直接調用創建匯出按鈕的函數
    createExportButton();
}); 
