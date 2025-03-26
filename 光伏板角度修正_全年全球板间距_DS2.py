import ephem
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os
import threading
from itertools import islice

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
# 全局锁用于结果累加和打印
result_lock = threading.Lock()
print_lock = threading.Lock()

# 配置参数
n = 10
angles_degrees = np.arange(0, 90.1, 0.1)
panel_spacings = np.arange(1, 2.1, 0.1)

# 预计算数据结构
vector1 = np.zeros((901, 3))
for angle in angles_degrees:
    angle_rad = math.radians(angle)
    vector1[int(angle*10)] = [0, -np.cos(angle_rad), np.sin(angle_rad)]

maxangle_tan = {}
for angle in angles_degrees:
    angle_rad = math.radians(angle)
    maxangle_tan[angle] = {}
    for ps in panel_spacings:
        denominator = ps - math.sin(angle_rad)
        maxangle_tan[angle][ps] = math.cos(angle_rad)/denominator if denominator !=0 else float('inf')

def date_chunk_generator(dates, chunk_size):
    """生成日期块迭代器"""
    it = iter(dates)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk

def process_date_chunk(lat_str, latitude, date_chunk, total_result):
    """处理日期块的核心计算逻辑"""
    obs = ephem.Observer()
    obs.lat = lat_str
    obs.lon = '0'
    is_southern = float(lat_str) < 0
    
    local_result = np.zeros_like(total_result)
    # print(f"开始处理日期块: {date_chunk}")
    for date_idx, date in date_chunk:
        try:
            sunrise, sunset = get_sunrise_sunset(obs, date)
            if sunrise is None or sunset is None:
                continue

            sunrise_jd = float(sunrise)
            sunset_jd = float(sunset)
            step = 10.0 / 1440  # 1分钟
            
            current_jd = sunrise_jd
            while current_jd <= sunset_jd:
                sun_alt, sun_az = calculate_sun_position(obs, ephem.Date(current_jd))
                if sun_alt < 0:
                    current_jd += step
                    continue

                # 计算太阳强度
                time_fraction = (current_jd - sunrise_jd) / (sunset_jd - sunrise_jd)
                intensity = math.sin(math.pi * time_fraction)

                # 计算太阳向量
                alt = float(sun_alt)
                az = float(sun_az)
                vector2 = np.array([
                    math.sin(az) * math.cos(alt),
                    math.cos(alt) * math.cos(az),
                    math.sin(alt)
                ])
                if is_southern:
                    vector2[1] = -vector2[1]

                # 计算所有参数组合
                for ps_idx, ps in enumerate(panel_spacings):
                    for angle_idx, angle in enumerate(angles_degrees):
                        dp = np.dot(vector1[int(angle*10)], vector2)
                        tan_alt = math.tan(alt)
                        threshold = maxangle_tan[angle][ps]
                        
                        if tan_alt >= threshold:
                            local_result[ps_idx, angle_idx] += intensity * dp * n
                        else:
                            k = (ps * math.sin(alt)) / math.cos(alt - math.radians(angle))
                            local_result[ps_idx, angle_idx] += intensity * dp * ((n-1)*k + 1)
                
                current_jd += step
        except Exception as e:
            with print_lock:
                print(f"纬度 {latitude} 日期 {date} 计算错误: {str(e)}")
    
    # 累加到全局结果
    with result_lock:
        total_result += local_result

def process_latitude(lat_info):
    """处理单个纬度的入口函数"""
    lat_str, latitude = lat_info
    with print_lock:
        print(f"\n开始计算纬度 {latitude}°...")
    
    # 生成日期范围（每个纬度单独生成）
    dates = []
    date = ephem.Date('2025/01/01')
    while date < ephem.Date('2026/01/01'):
        dates.append((0, date))  # 保持原有结构
        date += 1
    
    # 初始化结果存储
    total_result = np.zeros((len(panel_spacings), len(angles_degrees)))
    
    # 配置并行参数
    num_workers = os.cpu_count() or 1
    chunk_size = max(1, len(dates) // (num_workers * 2))
    
    # 创建线程池处理当前纬度的所有日期
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for chunk in date_chunk_generator(dates, chunk_size):
            futures.append(
                executor.submit(
                    process_date_chunk,
                    lat_str,
                    latitude,
                    chunk,
                    total_result
                )
            )
        
        # 等待所有任务完成
        for future in futures:
            future.result()
    
    # 查找每个板间距对应的最佳角度和结果
    best_angles = []
    best_results = []
    for ps_idx in range(len(panel_spacings)):
        max_index = np.argmax(total_result[ps_idx])
        best_angle = angles_degrees[max_index]
        best_angles.append(best_angle)
        best_results.append(total_result[ps_idx, max_index])
    
    # 查找全局最大结果以及对应的板间距和角度
    global_max_index = np.unravel_index(np.argmax(total_result), total_result.shape)
    best_spacing = panel_spacings[global_max_index[0]]
    best_angle_global = angles_degrees[global_max_index[1]]
    max_value = total_result[global_max_index]
    
    with print_lock:
        print(f"纬度 {latitude}° 完成 | 最佳角度列表: {[f'{angle:.1f}' for angle in best_angles]}")
    
    result = [latitude]
    for i in range(len(panel_spacings)):
        result.extend([best_angles[i], best_results[i]])
    result.extend([best_spacing, best_angle_global, max_value])
    
    return result

def main():
    # 生成纬度任务列表
    latitudes = [(str(lat), lat) for lat in range(-90, 91)]
    
    # 顺序处理每个纬度，每个纬度内部并行
    results = []
    for lat_info in latitudes:
        results.append(process_latitude(lat_info))
    
    # 构建列名
    columns = ['纬度']
    for ps in panel_spacings:
        columns.extend([f'最佳角度_板间距_{ps:.1f}', f'最佳结果_板间距_{ps:.1f}'])
    columns.extend(['最佳板间距', '最佳角度', '最大累积值'])
    
    # 保存结果
    df = pd.DataFrame(results, columns=columns)
    with pd.ExcelWriter('单纬度并行优化结果.xlsx') as writer:
        df.to_excel(writer, index=False)

if __name__ == '__main__':
    main()
