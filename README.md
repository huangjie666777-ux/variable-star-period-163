# Star Field Plate Solver + Differential Photometry

从零实现的星场定位(plate solving)、差分孔径测光与多夜周期识别后端:接收二维
FITS 图像与 ICRS 星表,不使用原 WCS 或在线服务。定位输出配对、残差、RMS 及
TAN 投影 WCS,可下载写入新 WCS 的 FITS;测光对 2-20 张同滤镜序列帧做差分孔径
测光,把透明度变化吸进逐帧零点,保留目标自身光变,输出光变曲线与 CSV;周期
分析接收同一目标的多批测光响应(每批附观测夜 ID),按误差平方倒数加权在频率
轴联合拟合各夜独立常数与共享正弦/余弦,输出周期谱、候选峰、采样窗与相位
折叠结果。

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
- app/samples.py — 多样本汇集:把多批测光响应(附 night_id)合并成一份
  多样本;仅 status=ok 且 MJD、星等有限、误差为正的帧参与,被排除帧
  保留原因与批次/帧来源。有效点须为 20-2000、至少 2 夜、时间跨度为正。
- app/periodogram.py — 联合加权周期搜索:每个频率联合拟合各夜独立常数
  与共享 sin/cos(不先减每夜均值),权重 1/err^2;功率为相对仅含夜常数
  模型的加权残差减少比例;相对时间(减去最早 MJD)保持精度,退化拟合
  标明原因。频率步长 <= 1/(5*跨度),最多 20000 频点,超限拒绝;最多
  返回 3 个间隔 >= 1/跨度 的局部峰,按功率排序、同值取较长周期;同时
  计算采样窗功率,提示边界峰与不足两周期的基线;最高峰只是候选,不是
  已确认周期。相位折叠以最早有效 MJD 为零点,给出每点相位、去夜零点
  星等、模型值与残差。
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
参考星(含排除原因);失败帧保留已知的 MJD/EXPTIME 与已算出的排除参考星。
响应同时携带 target_id,供周期分析校验同一目标。
GET /api/photometry/{photometry_id}/csv 下载同内容 CSV(含逗号的文件名
按 RFC 4180 加引号,不会错列)。

### POST /api/periodogram (multipart/form-data)

- params: JSON 字符串,例如:
  {"batches": [{"night_id": "night1", "photometry": <测光响应 JSON>}, ...],
   "period_min": 0.3, "period_max": 2.0}

所有批次须为同一 target_id;period 界须为正且递增(天)。返回频率轴、
功率谱、采样窗功率、候选峰(最多 3 个)、警告、退化拟合、被排除帧
(含原因与来源),以及按最强候选折叠的每点相位/去夜零点星等/模型值/
残差;没有可用候选时给出明确原因。GET /api/periodogram/{periodogram_id}/csv
下载与 JSON 一致的相位折叠 CSV(保留批次/帧来源)。

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

## 多夜周期示例

examples/multinight_demo.py 生成 3 夜 x 8 帧合成序列(目标按 0.62 天
正弦变化,每夜零点不同,透明度逐帧漂移),逐夜 POST /api/photometry,
再把三批响应 POST 到 /api/periodogram,打印候选周期并下载相位折叠 CSV
(examples/periodogram_phased.csv):

    .venv/bin/python examples/multinight_demo.py --url http://127.0.0.1:8152

恢复的候选周期与真值 0.62 天一致(仅为候选,需更多数据确认)。

## 测试范围

- tests/test_solver.py — 任意旋转/镜像/漏检/假源的定位;背向星、
  不可能 RMS、非法参数拒绝;下载 FITS 的 WCS 回算与观测头保留。
- tests/test_photometry.py — 变源 + 透明度漂移序列的差分测光:
  恢复星等跟踪真值、零点与透明度相关、常星光变曲线平坦;
  环星表顺序打乱后定位结果不变;PCOUNT/PSFREF 等观测头不被误删;
  缺 DATE-OBS 帧记为失败、其余帧继续;异常参考星被剔除并记录原因;
  非法参数(目标充当参考星、半径次序错误、帧数越界等)返回 422;
  定位失败帧保留已知 MJD/EXPTIME;含逗号文件名的 CSV 不错列。
- tests/test_periodogram.py — 多夜周期识别:恢复真实周期、频率步长与
  频点上限、峰间隔与排序、采样窗范围、相位零点与 CSV 一致、失败帧
  排除原因与来源、各类非法输入(点太少、单夜、周期界反序、目标不一致、
  频点超限)返回 422、常星无强峰与不足两周期基线警告。

## 范围说明

- 仅支持二维主 HDU 图像;不支持多扩展、压缩图像或畸变模型。
- 投影为标准 TAN(gnomonic),无视场畸变改正。
- 孔径测光为圆形孔径 + 圆环背景,不做 PSF 拟合;孔径内按整像素求和。
- 所有帧须为同一滤镜;输入图像不会被修改。
- 周期模型为单一正弦分量 + 各夜常数,不做多谐波或非正弦形状;
  最高峰是候选而非已确认周期;有效点限 20-2000、至少 2 夜、
  频点上限 20000。
- 结果保存在内存中,服务重启后 solve_id / photometry_id /
  periodogram_id 失效。
