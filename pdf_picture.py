import fitz  # PyMuPDF
import os
from PIL import Image
import imagehash
import io

# ========= 你只需要改这两个路径 =========
pdf_path = "D:\pdf抠图\声律启蒙2.pdf"    # ← 修改为你的 PDF 文件路径
output_dir = "E:\真桌面\声律启蒙"  # ← 修改为你想保存图片的目录
# ========================================

os.makedirs(output_dir, exist_ok=True)
doc = fitz.open(pdf_path)

seen_hashes = set()
img_count = 0
skip_count = 0

for page_index in range(len(doc)):
    page = doc[page_index]
    images = page.get_images(full=True)

    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        # 加载图像并计算感知哈希
        img_obj = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        hash_val = imagehash.phash(img_obj)  # 可换为 dhash、ahash 等

        if hash_val in seen_hashes:
            skip_count += 1
            continue  # 跳过重复图像
        seen_hashes.add(hash_val)

        # 保存图像
        img_filename = f"page{page_index+1}_img{img_index+1}.{image_ext}"
        img_path = os.path.join(output_dir, img_filename)
        with open(img_path, "wb") as f:
            f.write(image_bytes)
        img_count += 1

print(f"✅ 提取完成：共保存 {img_count} 张图像，跳过重复图像 {skip_count} 张")
