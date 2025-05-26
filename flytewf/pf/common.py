#!/usr/bin/env python3
import base64
import io
import os
import zipfile


def compress_directory_to_zip(input_dir: str, zip_path: str):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, input_dir)
                zipf.write(file_path, relative_path)


def extract_zip(zip_file_path: str, files_path: str):
    with zipfile.ZipFile(zip_file_path, 'r') as f:
        f.extractall(files_path)


def pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str
