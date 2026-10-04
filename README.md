# Star Field Plate Solver

从零实现的星场定位(plate solving)后端:接收二维 FITS 图像与 ICRS 星表,
不使用原 WCS 或在线服务,输出配对、残差、RMS 及 TAN 投影 WCS,并可下载
写入新 WCS 的 FITS 文件。

## 环境

- Python 3.10.12,FastAPI 0.115.12,Astropy 6.1.7,NumPy 2.2.6(见 requirements.lock)
- 所有命令使用 .venv/bin/python

## 模块划分(跨文件协作)

- app/extract.py — 中位数背景 + MAD 噪声估计;阈值成图、连通域检测、
  扣背景加权质心。排除非有限、饱和、边缘不完整源及孤立热像素;
  阈值 threshold_sigma 与饱和值 saturation 可按请求配置。
- app/match.py — 星表按 TAN(gnomonic)投影到切平面(角秒);拒绝距投影
  中心 >=90 度(背向)的星。三角形不变量(边长比,尺度/镜像无关)+ 变换参数
  投票做几何匹配,与列表顺序无关,支持旋转、平移、镜像;允许漏检、假源、
  视场外星;最终一对一贪心指派。搜索预算受 MATCH_TIME_BUDGET_S 限制,
  耗尽时明确报错。
- app/fit.py — 对一致配对最小二乘拟合仿射 pixel = A*tangent + b,
  MAD 迭代剔除离群并重拟合;要求 >=6 对且非共线(SVD 检查);
  最终 RMS(角秒)超过调用者 rms_max 则失败,不交付 WCS。
- app/wcsbuild.py — 由仿射逆构造 TAN WCS(CD 矩阵,度/像素);
  内部像素为零起点,FITS CRPIX 为一基参考点(已 +1 转换);
  清除旧 WCS 关键字(CRPIX/CRVAL/CD/PC/CDELT/CROTA/CTYPE/PV 等),
  保留像素值与其他观测头;不做任何畸变项。
- app/main.py — FastAPI 入口、参数校验与资源限制、结果存取。
- app/models.py — 请求/响应模型与服务器端限制
  (图像 <=4096x4096 像素、星表 <=2000 颗、FITS <=64MB 等)。

## API

### POST /api/solve (multipart/form-data)

- image: 二维 FITS 文件(主 HDU 含图像数据)
- params: JSON 字符串,例如:
  {"catalog": [{"id": "star001", "ra": 150.01, "dec": 20.02}, ...],
   "center_ra": 150.0, "center_dec": 20.0,
   "pixel_scale_min": 1.0, "pixel_scale_max": 1.6,
   "rms_max": 0.5, "threshold_sigma": 5.0, "saturation": 60000, "max_pairs": 200}

  pixel_scale_* 单位为角秒/像素;rms_max 单位角秒。
  非法参数(尺度范围倒置、星表 ID 重复、越界坐标、超限规模、
  背向投影中心的星)返回 422/413。

成功返回: solve_id、n_pairs、rms_arcsec、mirrored、
pairs(星表 ID、零起点像素质心、残差角秒)及 wcs(TAN 头关键字)。
配对不足、几何退化或 RMS 超标时返回 422 且不交付 WCS。

### GET /api/solve/{solve_id}/fits

下载写入新 WCS 的 FITS:像素值与其他观测头原样保留,旧 WCS 已清除。

## 运行与验证

    .venv/bin/python -m pytest tests -q          # 合成星场端到端测试
    .venv/bin/python -m compileall -q app tests  # 编译检查
    .venv/bin/python -m uvicorn app.main:app --port 8152

    curl -F "image=@field.fits" -F 'params={"catalog":[...],...}' \
         http://127.0.0.1:8152/api/solve
    curl -OJ http://127.0.0.1:8152/api/solve/<solve_id>/fits

## 测试范围

tests/test_solver.py 生成合成星场(高斯星点 + 噪声),覆盖:

- 任意旋转(35、80、123 度)与平移的求解;
- 镜像视场(mirrored=true);
- 漏检星(随机丢弃)、假源(视场外噪声峰)、孤立热像素排除;
- 拒绝背向投影中心的星表星;拒绝不可能达到的 RMS 门槛;
- 拒绝非法参数(尺度范围倒置);
- 下载 FITS 的 WCS 回算天区坐标与星表一致(<1 角秒),观测头保留。

## 范围说明

- 仅支持二维主 HDU 图像;不支持多扩展、压缩图像或畸变模型。
- 投影为标准 TAN(gnomonic),无视场畸变改正。
- 求解结果保存在内存中,服务重启后 solve_id 失效。
- 匹配预算固定为 20 秒;超大规模或极端密度星场可能因预算耗尽而失败。
