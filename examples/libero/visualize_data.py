import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image

# --- 1. 配置您的数据集和输出信息 (已根据您的要求配置) ---

# 数据集的根目录
dataset_root = Path("/media/brainsy/data/lebro/your_hf_username/libero")
# 数据集的数据文件所在的根目录
data_files_path = dataset_root / "data"

# 您希望保存可视化图片的输出目录
output_save_path = Path("/media/brainsy/data/lebro")

# --- 2. 检查路径并查找数据文件 ---

# 检查数据文件目录是否存在
if not data_files_path.is_dir():
    print(f"错误：在以下路径找不到数据文件目录：{data_files_path}")
    print("请确认您的路径是否正确。")
    exit()
else:
    print(f"成功找到数据文件目录：{data_files_path}")

# 使用 glob 递归查找所有的 .parquet 文件
episode_files = sorted(list(data_files_path.glob("**/*.parquet")))

if not episode_files:
    print(f"错误：在 '{data_files_path}' 目录中没有找到任何 .parquet 文件。")
    exit()

print(f"找到了 {len(episode_files)} 个 episode (.parquet) 文件。")

# --- 3. 可视化数据集内容 ---

# 在您指定的输出目录下，创建一个专门存放图片的文件夹
output_dir_image = output_save_path / "visualized_images/image"
output_dir_wrist = output_save_path / "visualized_images/wrist_image"
output_dir_image.mkdir(parents=True, exist_ok=True)
output_dir_wrist.mkdir(parents=True, exist_ok=True)

print(f"开始遍历数据集，并将可视化图片保存到: {output_save_path / 'visualized_images'}")

# 遍历找到的每一个 episode 文件
for episode_idx, episode_file in enumerate(episode_files):
    print(f"正在处理 Episode {episode_idx} (文件: {episode_file.name})...")
    try:
        episode_df = pd.read_parquet(episode_file)
    except Exception as e:
        print(f"  - 警告：读取文件 {episode_file.name} 时出错: {e}，已跳过。")
        continue

    if episode_idx == 0:
        print(f"  - 侦测到的列名: {episode_df.columns.to_list()}")

    for step_idx, step_data in episode_df.iterrows():
        try:
            image_dict = step_data["image"]
            wrist_image_dict = step_data["wrist_image"]

            # **修正**: 图像的相对路径应该从其对应的 .parquet 文件所在的目录开始拼接
            image_path = episode_file.parent / image_dict['path']
            wrist_image_path = episode_file.parent / wrist_image_dict['path']

            # 使用 Pillow 库从路径加载图像
            image_data = Image.open(image_path)
            wrist_image_data = Image.open(wrist_image_path)

            # 可视化并保存主摄像头图像
            plt.figure(figsize=(6, 6))
            plt.imshow(image_data)
            plt.axis("off")
            plt.title(f"Episode {episode_idx} / Step {step_idx} - Main")
            save_path_image = output_dir_image / f"ep{episode_idx:04d}_step{step_idx:04d}_main.png"
            plt.savefig(save_path_image)
            plt.close()

            # 可视化并保存手腕摄像头图像
            plt.figure(figsize=(6, 6))
            plt.imshow(wrist_image_data)
            plt.axis("off")
            plt.title(f"Episode {episode_idx} / Step {step_idx} - Wrist")
            save_path_wrist = output_dir_wrist / f"ep{episode_idx:04d}_step{step_idx:04d}_wrist.png"
            plt.savefig(save_path_wrist)
            plt.close()

        except KeyError as e:
            print(f"  - 警告：在图像字典中找不到键 'path'。字典内容: {image_dict}。错误: {e}")
            break
        except FileNotFoundError as e:
            print(f"  - 警告：找不到图像文件: {e}")
            break
        except Exception as e:
            print(f"  - 警告：处理 Episode {episode_idx}, Step {step_idx} 时发生未知错误: {e}")
            break


    print(f"  - 完成！已处理 {len(episode_df)} 个 steps。")

    # 为了快速演示，只可视化前2个 episode。如需处理全部数据，请注释掉或删除下面两行。
    if episode_idx >= 1:
        print("\n演示完成，已处理前2个 episodes。如需处理全部数据，请移除此处的 'break'。")
        break

print(f"\n可视化任务完成！图片已保存至 {output_save_path / 'visualized_images'} 目录。")
