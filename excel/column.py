import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. 读取保存的Excel文件
input_excel = 'models_info.xlsx'  # 也可以是 wrong_models_with_images.xlsx
output_image = 'outlier_distribution.png'

# 2. 读取数据
df = pd.read_excel(input_excel)

# 3. 获取Outlier_Ratio_Percentage列
outlier_percentages = df['Outlier_Ratio_Percentage']

# 4. 定义区间 (0-10%、10-20%、...、90-100%)
bins = [i for i in range(0, 110, 10)]  # 0,10,20,...100

# 5. 使用pandas.cut分组
labels = [f"{bins[i]}%-{bins[i+1]}%" for i in range(len(bins)-1)]
df['Outlier_Percentage_Bin'] = pd.cut(outlier_percentages, bins=bins, labels=labels, right=False)

# 6. 统计每个区间的数量
bin_counts = df['Outlier_Percentage_Bin'].value_counts().sort_index()

# 7. 绘制柱状图
plt.figure(figsize=(10, 6))
bin_counts.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Outlier Ratio Percentage Distribution')
plt.xlabel('Outlier Ratio Range')
plt.ylabel('Number of Models')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 8. 保存图片
plt.tight_layout()
plt.savefig(output_image, dpi=300)
plt.show()

print(f"柱状图已保存为：{output_image}")