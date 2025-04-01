import csv
import os.path as osp
import re
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
# 指定文件路径
file_path = osp.join(project_root,"evaluation/result/score_contrast_ORSSD/result.txt")
# 指定输出的 CSV 文件路径
output_csv_path = osp.join(project_root,"evaluation/result/extracted_all_data.csv")

# 初始化表头和结果列表
headers = ['Dataset', 'Method']
results = []

try:
    # 打开文件并逐行读取
    with open(file_path, 'r') as file:
        for line in file:
            # 使用正则表达式匹配包含任意数据集名和模型名的行
            match = re.search(r'#\s*\[\s*.*\s*Dataset\]\s*\[\s*.*\s*Method\]\s*#', line)
            if match:
                # print(f"匹配到的行: {line}")
                # 提取数据集名
                dataset_match = re.search(r'\[\s*(.*?)\s*Dataset\]', line)
                if dataset_match:
                    dataset = dataset_match.group(1).strip()
                    # print(f"提取的数据集名: {dataset}")
                else:
                    print("未提取到数据集名")
                    continue
                # 提取方法名
                method_match = re.search(r'#(.*)Dataset\] \[(.*) Method\]#', line)
                if method_match:
                    method = method_match.group(2).strip()
                    # print(f"提取的方法名: {method}")
                else:
                    print("未提取到方法名")
                    continue

                # 提取所有指标数据
                metric_matches = re.findall(r'\[([\d.]+)\s+([^]]+)\]', line)
                data = {'Dataset': dataset, 'Method': method}
                for value, metric_name in metric_matches:
                    if metric_name not in headers:
                        headers.append(metric_name)
                    data[metric_name] = value

                results.append(data)
            else:
                print(f"未匹配到符合条件的行: {line}")

    # 将结果写入 CSV 文件
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        # 写入表头
        writer.writeheader()
        # 写入数据行
        for result in results:
            writer.writerow(result)

    print(f"数据已成功保存到 {output_csv_path}")

except FileNotFoundError:
    print(f"文件 {file_path} 未找到。")
except Exception as e:
    print(f"发生错误: {e}")
