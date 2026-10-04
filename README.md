# Star Field Plate Solver + Differential Photometry

从零实现的星场定位(plate solving)与差分孔径测光后端:接收二维 FITS 图像与
ICRS 星表,不使用原 WCS 或在线服务。定位输出配对、残差、RMS 及 TAN 投影 WCS,
可下载写入新 WCS 的 FITS;测光对 2-20 张同滤镜序列帧做差分孔径测光,把透明度
变化吸进逐帧零点,保留目标自身光变,输出光变曲线与 CSV。

## 环境

- Python 3.10.12,FastAPI 0.115.12,Astropy 6.1.7,NumPy 2.2.6(见 requirements.lock)
- 所有命令使用 .venv/bin/python

## 模块划分(跨文件协作)

- app/extract.py — 中位数背景 + MAD 噪声估计;阈值成图、连通域检测、
  扣背景加权质心。排除非有限、饱和、边缘不完整源及孤立热像素。
- app/match.py — 星表按 TAN(gnomonic)投影到切平面(角秒);拒绝距投影
  中心 >=90 度(背向)的星。三角形不变量 + 变换参数投票做几何匹配。
  三角形星表池按"距星表质心最近 + 坐标确定性次序"选取,与调用方传入的
  星表顺序无关;投票键包含尺度、旋转角与平移,避免不同旋转的三角形
  假设在同一票箱里被平均成错误变换。
- app/fit.py — 对一致配对最小二乘拟合仿射 pixel = A*tangent + b,
  MAD 迭代剔除离群;要求 >=6 对且非共线;RMS 超过 rms_max 则失败。
- app/wcsbuild.py — 由仿射逆构造 TAN WCS(CD 矩阵,度/像素;CRPIX 一基)。
  仅按精确编号模式清除旧 WCS 关键字(CD1_1、PC1_1、CDELT1、CTYPE1 等),
  不会误删 PCOUNT、PSFREF 等共享前缀的观测头;像素值与其他头原样保留。
- app/pipeline.py — 定位流水线(提取→投影→匹配→拟合),/api/solve 与
  /api/photometry 每帧共用同一代码路径。
- app/photometry.py — 孔径测光:像素中心落入孔径求和;背景环内排除其他
  已检源后以中位数稳健估计局部背景;噪声模型合并泊松、读出与背景估计
  三项。非有限、饱和、越界、拥挤、非正通量分别以原因标记,不给星等。
- app/calibrate.py — 逐帧零点:zp_i = mag_i + 2.5*log10(rate_i),中位数 +
  MAD 稳健剔除异常参考星(不少于 3 颗,否则该帧失败);目标星等与误差
  由通量误差与零点误差传播合成。目标星不参与定标。
- app/main.py — FastAPI 入口、参数校验、结果存取、CSV 生成。
- app/models.py — 请求/响应模型与服务器端限制。

## API

### POST /api/solve (multipart/form-data)

- image: 二维 FITS 文件(主 HDU 含图像数据)
- params: JSON 字符串,例如:
  {"catalog": [{"id": "star001", "ra": 150.01, "dec": 20.02}, ...],
   "center_ra": 150.0, "center_dec": 20.0,
   "pixel_scale_min": 1.0, "pixel_scale_max": 1.6,
   "rms_max": 0.5, "threshold_sigma": 5.0, "saturation": 60000, "max_pairs": 200}

成功返回 solve_id、n_pairs、rms_arcsec、mirrored、pairs 及 wcs。
GET /api/solve/{solve_id}/fits 下载写入新 WCS 的 FITS。

### POST /api/photometry (multipart/form-data)

- images: 2-20 张已校正(平场/暗场等)同滤镜二维 FITS,需含有效
  DATE-OBS(UTC)与正 EXPTIME;时间戳取 UTC 曝光中点。
- params: JSON 字符串,例如:
  {"solve": {同上 SolveParams},
   "target_id": "varstar",
   "references": [{"id": "ref0", "mag": 12.0}, ...至少 3 颗...],
   "aperture_radius": 4.0, "annulus_inner": 8.0, "annulus_outer": 13.0,
   "gain": 2.5, "read_noise": 4.0}

校验:目标与参考星 ID 必须在星表中、参考星 ID 唯一、目标不得充当参考星、
半径须满足 aperture < annulus_inner < annulus_outer、gain>0、read_noise>=0。
非法输入返回 422。

每帧独立定位与测光;单帧失败(定位失败、头缺失、可用参考星不足 3 颗等)
保留失败记录与原因,其余帧继续。返回按曝光中点排序的帧列表:MJD、
EXPTIME、扣背景通量率及误差、目标星等及误差、零点及误差、采用与排除的
参考星(含排除原因)。GET /api/photometry/{photometry_id}/csv 下载同内容 CSV。

## 运行与验证

    .venv/bin/python -m pytest tests -q          # 定位 + 测光端到端测试
    .venv/bin/python -m compileall -q app tests  # 编译检查
    .venv/bin/python -m uvicorn app.main:app --port 8152

    curl -F "image=@field.fits" -F 'params={"catalog":[...],...}' \
         http://127.0.0.1:8152/api/solve
    curl -OJ http://127.0.0.1:8152/api/solve/<solve_id>/fits

    curl -F 'params=<params.json;type=application/json' \
         -F "images=@night1_00.fits" -F "images=@night1_01.fits" ... \
         http://127.0.0.1:8152/api/photometry
    curl -OJ http://127.0.0.1:8152/api/photometry/<photometry_id>/csv

## 光变示例

examples/lightcurve_demo.py 生成 6 帧合成序列(目标按 +-0.3 mag 变化,
透明度按 +-0.25 mag 漂移),POST 到运行中的服务并打印恢复的光变曲线,
CSV 存为 examples/lightcurve.csv:

    .venv/bin/python examples/lightcurve_demo.py --url http://127.0.0.1:8152

恢复的目标星等与真值一致(透明度漂移由逐帧零点吸收)。

## 测试范围

- tests/test_solver.py — 任意旋转/镜像/漏检/假源的定位;背向星、
  不可能 RMS、非法参数拒绝;下载 FITS 的 WCS 回算与观测头保留。
- tests/test_photometry.py — 变源 + 透明度漂移序列的差分测光:
  恢复星等跟踪真值、零点与透明度相关、常星光变曲线平坦;
  环星表顺序打乱后定位结果不变;PCOUNT/PSFREF 等观测头不被误删;
  缺 DATE-OBS 帧记为失败、其余帧继续;异常参考星被剔除并记录原因;
  非法参数(目标充当参考星、半径次序错误、帧数越界等)返回 422。

## 范围说明

- 仅支持二维主 HDU 图像;不支持多扩展、压缩图像或畸变模型。
- 投影为标准 TAN(gnomonic),无视场畸变改正。
- 孔径测光为圆形孔径 + 圆环背景,不做 PSF 拟合;孔径内按整像素求和。
- 所有帧须为同一滤镜;输入图像不会被修改。
- 结果保存在内存中,服务重启后 solve_id / photometry_id 失效。

