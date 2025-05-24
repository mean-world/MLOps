from PIL import Image
import os


def resize(path: str, file_name: str, scale: float):
    path = path + "/"
    original_image = Image.open(path + file_name)
    original_width, original_height = original_image.size

    target_width = int(original_width * scale)
    target_height = int(original_height * scale)
    low_res_image_lanczos = original_image.resize(
        (target_width, target_height), resample=Image.BICUBIC
    )

    file_path = "downsample_0.25/"
    new_name = file_path + file_name[:-4] + ".png"
    low_res_image_lanczos.save(new_name)


# resize("image/0.jpg", 0.25)
path = os.listdir("image")
for i in path:
    resize("image", i, 0.25)
