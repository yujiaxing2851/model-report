import numpy as np
import pandas as pd
import os
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage

# 参数设置
distance_file = '../datas/distances_BASF_860_LEVEL1-3.npy'
model_id_file = '../datas/object_indices.npy'
txt_file = '../datas/mesh_index_map.txt'

threshold_distance = 0.03
outlier_ratio_threshold = 0.5
base_url = 'https://www.kk.com/searchid=?'            # 模型定位链接的基础部分

error_pointcloud_folder = 'error_pointcloud_images'
error_model_folder = 'error_model_images'
comparison_folder = 'comparison_images'

output_excel = 'wrong_models_with_images.xlsx'

# 1. 加载 npy
distances = np.load(distance_file)
model_ids = np.load(model_id_file)

if distances.shape[0] != model_ids.shape[0]:
    raise ValueError("距离文件和模型编号文件中的数据数量不一致！")

# 2. 分组统计
model_stats = {}
for distance, mid in zip(distances, model_ids):
    if mid not in model_stats:
        model_stats[mid] = {"total": 0, "outlier_count": 0}
    model_stats[mid]["total"] += 1
    if distance > threshold_distance:
        model_stats[mid]["outlier_count"] += 1

# 3. 整理数据
output_data = []
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
        'Is_Wrong': is_wrong,
        'Threshold': threshold_distance,
        'Outlier_Ratio_Percentage': ratio * 100,
    }
    output_data.append(model_info)

# 4. 加载txt
model_name_mapping = {}
with open(txt_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            try:
                key = int(parts[0])
                model_name = parts[1]
                model_name_mapping[key] = model_name
            except ValueError:
                continue

# 5. 转DataFrame，添加名字和图片路径（先写路径）
df_models = pd.DataFrame(output_data)
df_models['Model_Name'] = df_models['Model_ID'].map(model_name_mapping)

def get_image_path(folder, model_id):
    file_path = os.path.join(folder, f'{model_id}.png')
    if os.path.exists(file_path):
        return file_path
    else:
        return None

df_models['Error_PointCloud_Image'] = df_models['Model_ID'].apply(lambda x: get_image_path(error_pointcloud_folder, x))
df_models['Error_Model_Image'] = df_models['Model_ID'].apply(lambda x: get_image_path(error_model_folder, x))
df_models['Comparison_Image'] = df_models['Model_ID'].apply(lambda x: get_image_path(comparison_folder, x))

# 6. 只保留错误模型
df_wrong = df_models[df_models['Is_Wrong'] == True]
df_wrong['WrongLocation'] = base_url + df_wrong['Model_Name'].astype(str)
# 7. 先保存基础信息到Excel（没有图片）
basic_columns = [
    'Model_ID', 'Model_Name', 'Total_Points', 'Outlier_Points', 
    'Outlier_Ratio', 'Outlier_Ratio_Percentage', 'Threshold',
    'Error_PointCloud_Image', 'Error_Model_Image', 'Comparison_Image', 'WrongLocation'
]
df_wrong.to_excel(output_excel, index=False, columns=basic_columns)

# 8. 插入真正的图片到Excel
wb = load_workbook(output_excel)
ws = wb.active

# 图片列对应的Excel列
img_col_mapping = {
    'Error_PointCloud_Image': 'H',  # Excel列H
    'Error_Model_Image': 'I',        # Excel列I
    'Comparison_Image': 'J',         # Excel列J
}

# 从第二行开始（第一行是表头）
for idx, row in enumerate(df_wrong.itertuples(), start=2):
    for col_name, excel_col in img_col_mapping.items():
        img_path = getattr(row, col_name)
        if img_path and os.path.exists(img_path):
            img = XLImage(img_path)
            # 设置图片大小（可以根据需求调整）
            img.width = 100
            img.height = 100
            cell = f"{excel_col}{idx}"
            ws.add_image(img, cell)

# 最后保存
wb.save(output_excel)

df_models.to_excel('models_info.xlsx', index=False)
df_wrong.to_excel('wrong_models.xlsx', index=False)

print(f"最终带图片的Excel保存到：{output_excel}")
