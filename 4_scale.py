import os
from PIL import Image
import torch
import numpy as np

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# ========== 配置区域 ==========
input_folder = r"E:\test1"        # 输入文件夹路径
output_folder = r"E:\return1"           # 输出文件夹路径
scale_factor = 4                        # 放大倍数
model_path = r"E:\模型\图片分辨率4倍\realesr-animevideov3.pth"  # 模型权重路径
# ==============================

os.makedirs(output_folder, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 构建与权重匹配的模型结构
model = SRVGGNetCompact(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_conv=16,
    upscale=scale_factor
)

upsampler = RealESRGANer(
    scale=scale_factor,
    model_path=model_path,
    model=model,
    device=device,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
)

image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

print(f"共发现 {len(image_files)} 张图像，开始处理...")

for i, filename in enumerate(image_files, 1):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    try:
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img)

        # 放大图像
        output, _ = upsampler.enhance(img_np, outscale=scale_factor)
        sr_image = Image.fromarray(output)

        sr_image.save(output_path)
        print(f"[{i}/{len(image_files)}] 已处理：{filename}")
    except Exception as e:
        print(f"[!] 处理失败：{filename}，原因：{e}")

print("全部处理完成！结果保存在输出文件夹中。")
