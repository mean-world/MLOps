from PIL import Image

# 開啟原始圖片
original_image = Image.open("image/0.jpg")
original_width, original_height = original_image.size

# 指定縮小比例 (例如縮小到原來的 1/2)
scale_factor = 0.25

# 計算目標寬度和高度
target_width = int(original_width * scale_factor)
target_height = int(original_height * scale_factor)

# 使用 Lanczos 插值調整大小
low_res_image_lanczos = original_image.resize(
    (target_width, target_height), resample=Image.BICUBIC
)
# low_res_image_lanczos.save("low_res_lanczos.png")


low_res_image_lanczos.show()
