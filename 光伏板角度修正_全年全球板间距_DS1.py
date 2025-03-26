import ephem
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os
import threading
from tqdm import tqdm  # 导入 tqdm 库

# 初始化全局锁用于打印同步
print_lock = threading.Lock()

# 板数量
n = 10
# 预计算所有角度的向量数组
angles_degrees = np.arange(0, 90.1, 0.1)
# 定义板间距范围
panel_spacings = np.arange(1, 2.1, 0.1)

# 预计算 vector1
vector1 = np.zeros((901, 3))
for angle in angles_degrees:
    angle_rad = math.radians(angle)
    vector1[int(angle * 10)] = [0, -np.cos(angle_rad), np.sin(angle_rad)]

maxangle_tan = {}
for angle in angles_degrees:
    angle_rad = math.radians(angle)
    maxangle_tan[angle] = {}
    for panel_spacing in panel_spacings:
        denominator = panel_spacing - math.sin(angle_rad)
        maxangle_tan[angle][panel_spacing] = math.cos(angle_rad) / denominator if denominator != 0 else float('inf')

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

# 初始化日期范围
start_date = '2025/01/01'
end_date = '2026/01/01'
dates = []
date = ephem.Date(start_date)
date_index = 0
while date < ephem.Date(end_date):
    dates.append((date_index, date))
    date += 1
    date_index += 1

def process_latitude(lat_info):
    lat_str, latitude = lat_info
    lon_str = '0'
    total_result = np.zeros((len(panel_spacings), len(angles_degrees)))
    is_southern_hemisphere = float(lat_str) < 0

    for date_index, date in dates:
        obs = ephem.Observer()
        obs.lat = lat_str
        obs.lon = lon_str

        sunrise, sunset = get_sunrise_sunset(obs, date)
        if sunrise is None or sunset is None:
            continue

        sunrise_jd = float(sunrise)
        sunset_jd = float(sunset)
        step = 1.0 / 24 / 60  # 1分钟步长
        current_jd = sunrise_jd

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

            if is_southern_hemisphere:
                vector2[1] = -vector2[1]

            # 向量化计算
            for panel_spacing_idx, ps in enumerate(panel_spacings):
                for angle_idx, angle in enumerate(angles_degrees):
                    dp = np.dot(vector1[int(angle * 10)], vector2)
                    tan_alt = math.tan(alt)
                    threshold = maxangle_tan[angle][ps]

                    if tan_alt >= threshold:
                        total_result[panel_spacing_idx, angle_idx] += intensity * dp * n
                    else:
                        k = (ps * math.sin(alt)) / math.cos(alt - math.radians(angle))
                        total_result[panel_spacing_idx, angle_idx] += intensity * dp * ((n - 1) * k + 1)
            current_jd += step

    max_index = np.unravel_index(total_result.argmax(), total_result.shape)
    best_angle = angles_degrees[max_index[1]]
    best_spacing = panel_spacings[max_index[0]]
    max_value = total_result[max_index]

    with print_lock:
        print(f"纬度 {latitude} 完成 | 最佳角度: {best_angle:.1f}° 间距: {best_spacing:.1f}m 能量: {max_value:.2f}")

    return [latitude, best_angle, best_spacing, max_value]

def main():
    # 生成纬度任务列表
    latitudes = [(str(lat), lat) for lat in range(-90, 91)]

    # 获取CPU核心数（包含超线程）
    max_workers = os.cpu_count() or 1

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 移除整体进度条
        results = list(executor.map(process_latitude, latitudes))

    # 保存结果
    df = pd.DataFrame(results, columns=['纬度', '最佳角度', '最佳板间距', '最大累积值'])
    with pd.ExcelWriter('优化后光伏板角度数据_多线程.xlsx') as writer:
        df.to_excel(writer, index=False)

if __name__ == '__main__':
    main()
