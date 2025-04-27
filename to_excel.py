import numpy as np
import pandas as pd

# 参数设置
distance_file = 'output/distances_indices.npy'              # 距离信息文件路径
model_id_file = 'output/object_indices.npy'              # 模型编号文件路径
txt_file = 'output/mesh_index_map.txt'                 # 存放模型名称的txt文件路径
threshold_distance = 0.03                            # 距离阈值（单位：米，3cm）
outlier_ratio_threshold = 0.5                        # 异常点比例阈值（大于50%视为异常模型）
base_url = 'https://www.kk.com/searchid=?'            # 模型定位链接的基础部分

# 1. 加载 npy 数据
distances = np.load(distance_file)   # distances: shape (2000,), dtype=float64
model_ids = np.load(model_id_file)     # model_ids: shape (2000,), dtype=int64

if distances.shape[0] != model_ids.shape[0]:
    raise ValueError("距离文件和模型编号文件中的数据数量不一致！")

# 2. 按模型编号分组统计
model_stats = {}
for distance, mid in zip(distances, model_ids):
    if mid not in model_stats:
        model_stats[mid] = {"total": 0, "outlier_count": 0}
    model_stats[mid]["total"] += 1
    if distance > threshold_distance:
        model_stats[mid]["outlier_count"] += 1

# 3. 判断哪些模型错误，并整理输出信息
output_data = []
wrong_models = []  # 用来存储错误模型的 model id

for mid, stats in model_stats.items():
    total = stats["total"]
    outlier_count = stats["outlier_count"]
    ratio = outlier_count / total
    is_wrong = ratio > outlier_ratio_threshold
    model_info = {
        'Model_ID': mid,
        'Total_Points': total,
        'Outlier_Points': outlier_count,
        'Outlier_Ratio': ratio,
        'Is_Wrong': is_wrong
    }
    output_data.append(model_info)
    if is_wrong:
        wrong_models.append(mid)

print('完成统计')
# 整理所有模型的统计信息为 DataFrame
df_models = pd.DataFrame(output_data)

# 单独提取错误模型信息
df_wrong = df_models[df_models['Is_Wrong'] == True]

# 4. 读取 txt 文件，构建序号与模型名称映射字典
# txt 文件中每行格式：序号 模型名称（中间以空格分隔）
model_name_mapping = {}
with open(txt_file, 'r', encoding='utf-8') as f:
    for line in f:
        # 仅以第一个空格切分，剩余部分作为名称（有些名称中可能包含空格）
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            try:
                key = int(parts[0])
                model_name = parts[1]
                model_name_mapping[key] = model_name
            except ValueError:
                # 如果无法转换为整型则跳过该行
                continue

# 5. 根据 DataFrame 中的 Model_ID 添加对应模型名称信息
# 假设 DataFrame 中的 'Model_ID' 和 txt 文件中的序号是对应的
df_models['Model_Name'] = df_models['Model_ID'].map(model_name_mapping)
df_wrong['Model_Name'] = df_wrong['Model_ID'].map(model_name_mapping)


df_wrong['WrongLocation'] = base_url + df_wrong['Model_Name'].astype(str)

# 6. 保存 Excel 文件
# 保存完整模型统计信息以及错误模型信息
df_models.to_excel('models_info.xlsx', index=False)
df_wrong.to_excel('wrong_models.xlsx', index=False)

print("保存完毕！共判定出 %d 个错误模型。" % len(df_wrong))