import ephem
import numpy as np
import math
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# 板数量
n = 10
# 预计算所有角度的向量数组
angles_degrees = np.arange(0, 90.1, 0.1)
# 定义板间距范围
panel_spacings = np.arange(1, 2.1, 0.1)

# 纬度范围
latitudes = np.arange(-90, 91)

# 预计算 vector1
vector1 = np.zeros((901, 3))
for angle in angles_degrees:
    angle_rad = math.radians(angle)
    # Bug 修复：将计算得到的向量赋值给 vector1 数组的对应位置
    vector1[int(angle * 10)] = [0, -np.cos(angle_rad), np.sin(angle_rad)]

maxangle_tan = {}
for angle in angles_degrees:
    angle_rad = math.radians(angle)
    maxangle_tan[angle] = {}
    for panel_spacing in panel_spacings:
        denominator = panel_spacing - math.sin(angle_rad)
        if denominator == 0:
            # 处理除数为零的情况，可以将结果设为无穷大或其他合适的值
            maxangle_tan[angle][panel_spacing] = float('inf')
        else:
            maxangle_tan[angle][panel_spacing] = math.cos(angle_rad) / denominator

def calculate_sun_position(obs, date_obj):
    """计算给定观测时间和地点的太阳高度角和方位角"""
    obs.date = date_obj
    sun = ephem.Sun()
    sun.compute(obs)
    return sun.alt, sun.az

def get_sunrise_sunset(obs, date_obj):
    """计算当天的日出日落时间"""
    obs.date = date_obj
    sun = ephem.Sun()
    try:
        return obs.next_rising(sun), obs.next_setting(sun)
    except ephem.AlwaysUpError:
        date_tuple = ephem.Date(date_obj).tuple()
        start_date = ephem.Date(date_tuple[:3] + (0, 0, 0))
        return start_date, start_date + 1
    except ephem.NeverUpError:
        return None, None


start_date = '2025/01/01'
end_date = '2026/01/01'

# 生成日期范围
dates = []
date = ephem.Date(start_date)
date_index = 0
while date < ephem.Date(end_date):
    dates.append((date_index, date))
    date += 1
    date_index += 1

# 预生成纬度字符串
latitudes = [(str(lat), lat) for lat in range(-90, 91)]

# 准备结果存储
best_results = []

# 遍历每个纬度
for lat_str, latitude in latitudes:
    print(f"\n开始计算纬度 {latitude} 度...")
    lon_str = '0'
    # 初始化每个纬度的总结果数组
    total_result = np.zeros((len(panel_spacings), len(angles_degrees)))

    # 遍历每个日期
    for date_index, date in dates:
        print(f"开始计算日期 {date}...")
        obs = ephem.Observer()
        obs.lat = lat_str
        obs.lon = lon_str

        lat = int(float(lat_str))

        total_result_day = np.zeros((len(panel_spacings), len(angles_degrees)))
        sunrise, sunset = get_sunrise_sunset(obs, date)

        if sunrise is None or sunset is None:
            continue

        sunrise_jd = float(sunrise)
        sunset_jd = float(sunset)

        # 计算时间步长（1分钟）
        step = 1.0 / 24 / 60
        current_jd = sunrise_jd

        # 判断是否为南半球
        is_southern_hemisphere = float(lat_str) < 0

        while current_jd <= sunset_jd:
            sun_alt, sun_az = calculate_sun_position(obs, ephem.Date(current_jd))
            if sun_alt < 0:
                current_jd += step
                continue

            # 计算太阳强度
            time_fraction = (current_jd - sunrise_jd) / (sunset_jd - sunrise_jd)
            intensity = math.sin(math.pi * time_fraction)

            # 计算太阳方向向量
            alt = float(sun_alt)
            az = float(sun_az)
            vector2 = np.array([
                math.sin(az) * math.cos(alt),
                math.cos(alt) * math.cos(az),
                math.sin(alt)
            ])

            # 如果是南半球，调整向量方向
            if is_southern_hemisphere:
                vector2[1] = -vector2[1]

            for panel_spacing_index, panel_spacing in enumerate(panel_spacings):
                for angle_index, angle in enumerate(angles_degrees):
                    dot_product = np.dot(vector1[int(angle * 10)], vector2)
                    if math.tan(alt) >= maxangle_tan[angle][panel_spacing]:
                        # 无遮挡关系直接用 vector1 与 vector2 点乘
                        total_result_day[panel_spacing_index, angle_index] += intensity * dot_product * n
                    else:
                        # 遮挡比例k
                        k = (panel_spacing * math.sin(alt)) / math.cos(alt - math.radians(angle))
                        # 有遮挡关系，需要考虑遮挡板的影响
                        total_result_day[panel_spacing_index, angle_index] += intensity * dot_product * ((n - 1) * k + 1)
            current_jd += step

        # 将当天的结果累加到总结果中
        total_result += total_result_day

    # 完成所有日期的计算后按照纬度寻找最大累积值对应的角度和间距
    max_cumulative_value = total_result.max()
    max_index = np.unravel_index(total_result.argmax(), total_result.shape)
    best_spacing = panel_spacings[max_index[0]]
    best_angle = angles_degrees[max_index[1]]

    # 每个维度计算完成后输出结果
    print(f"纬度 {latitude} 度的最佳角度是 {best_angle} 度，最佳板间距是 {best_spacing}，最大累积值是 {max_cumulative_value}")

    best_results.append([
        latitude,
        best_angle,
        best_spacing,
        max_cumulative_value
    ])

# 创建DataFrame并保存
df_best = pd.DataFrame(
    best_results,
    columns=['纬度', '最佳角度', '最佳板间距', '最大累积值']
)

with pd.ExcelWriter('优化后光伏板角度数据.xlsx') as writer:
    df_best.to_excel(writer, sheet_name='最佳角度', index=False)