# app/02_image_basic_processing.py

"""
실제 이미지 파일을 OpenCV로 읽고 기본 전처리를 수행하는 예제입니다.

실습 내용:
1. 이미지 읽기
2. 이미지 크기 확인
3. 리사이즈
4. 중앙 영역 자르기
5. 흑백 변환
6. 결과 저장

주의:
- sample.jpg 파일은 /workspace/data/images/sample.jpg 위치에 있어야 합니다.
- Windows 기준 위치는 C:\\dev\\opencv-docker-lab\\data\\images\\sample.jpg 입니다.
"""

import os
import cv2


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    print("=== 실제 이미지 기본 처리 실습 시작 ===")

    input_path = "/workspace/data/images/sample.jpg"
    output_dir = "/workspace/outputs"

    ensure_dir(output_dir)

    # 1. 이미지 읽기
    image = cv2.imread(input_path)

    # 2. 이미지 읽기 실패 확인
    if image is None:
        raise FileNotFoundError(
            f"이미지를 읽을 수 없습니다: {input_path}\n"
            "확인 사항:\n"
            "1) data/images 폴더에 sample.jpg가 있는지 확인하세요.\n"
            "2) 파일명이 sample.jpg와 정확히 일치하는지 확인하세요.\n"
            "3) 확장자가 .jpeg, .png가 아닌지 확인하세요."
        )

    # 3. 이미지 크기 확인
    height, width, channels = image.shape

    print(f"원본 이미지 크기: width={width}, height={height}, channels={channels}")

    # 4. 리사이즈
    resized_width = 640
    resized_height = 480

    resized = cv2.resize(image, (resized_width, resized_height))

    resized_path = os.path.join(output_dir, "sample_resized_640x480.jpg")
    cv2.imwrite(resized_path, resized)

    print(f"리사이즈 결과 저장: {resized_path}")

    # 5. 중앙 영역 자르기
    # OpenCV/NumPy 배열 인덱싱은 [y1:y2, x1:x2] 순서입니다.
    crop_size = 300

    center_x = width // 2
    center_y = height // 2

    x1 = max(center_x - crop_size // 2, 0)
    y1 = max(center_y - crop_size // 2, 0)
    x2 = min(center_x + crop_size // 2, width)
    y2 = min(center_y + crop_size // 2, height)

    cropped = image[y1:y2, x1:x2]

    cropped_path = os.path.join(output_dir, "sample_center_crop.jpg")
    cv2.imwrite(cropped_path, cropped)

    print(f"중앙 자르기 결과 저장: {cropped_path}")
    print(f"자른 영역 좌표: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # 6. 흑백 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_path = os.path.join(output_dir, "sample_gray.jpg")
    cv2.imwrite(gray_path, gray)

    print(f"흑백 변환 결과 저장: {gray_path}")

    print("=== 실제 이미지 기본 처리 실습 완료 ===")


if __name__ == "__main__":
    main()