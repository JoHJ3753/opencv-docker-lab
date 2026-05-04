# app/03_color_space_and_mask.py

"""
OpenCV 색상 공간 변환과 색상 기반 객체 검출 기초 실습

실습 내용:
1. BGR 테스트 이미지 생성
2. BGR 이미지를 RGB / HSV로 변환
3. 특정 색상 영역 마스크 생성
4. 마스크 결과 저장
5. 색상 검출 결과 저장

주의:
- OpenCV 기본 색상 순서는 RGB가 아니라 BGR입니다.
- HSV에서 H 범위는 0~179입니다.
- cv2.imshow()는 사용하지 않고 결과 파일을 outputs 폴더에 저장합니다.
"""

import os
import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_image(path: str, image) -> None:
    success = cv2.imwrite(path, image)
    if not success:
        raise IOError(f"이미지 저장 실패: {path}")


def main():
    print("=== 색상 공간 변환 및 색상 검출 실습 시작 ===")

    output_dir = "/workspace/outputs"
    ensure_dir(output_dir)

    # 1. 테스트 이미지 생성
    height = 400
    width = 600

    # 흰색 배경 이미지 생성
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    # OpenCV 색상은 BGR 순서입니다.
    blue_bgr = (255, 0, 0)
    green_bgr = (0, 255, 0)
    red_bgr = (0, 0, 255)
    yellow_bgr = (0, 255, 255)
    black_bgr = (0, 0, 0)

    # 파란 사각형
    cv2.rectangle(
        image,
        pt1=(50, 60),
        pt2=(220, 180),
        color=blue_bgr,
        thickness=-1,
    )

    # 초록 원
    cv2.circle(
        image,
        center=(420, 120),
        radius=70,
        color=green_bgr,
        thickness=-1,
    )

    # 빨간 사각형
    cv2.rectangle(
        image,
        pt1=(70, 250),
        pt2=(250, 350),
        color=red_bgr,
        thickness=-1,
    )

    # 노란 원
    cv2.circle(
        image,
        center=(430, 300),
        radius=60,
        color=yellow_bgr,
        thickness=-1,
    )

    # 라벨 표시
    cv2.putText(image, "BLUE", (95, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, black_bgr, 2)
    cv2.putText(image, "GREEN", (365, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, black_bgr, 2)
    cv2.putText(image, "RED", (125, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.8, black_bgr, 2)
    cv2.putText(image, "YELLOW", (370, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, black_bgr, 2)

    original_path = os.path.join(output_dir, "03_original_bgr.png")
    save_image(original_path, image)
    print(f"원본 BGR 테스트 이미지 저장: {original_path}")

    # 2. BGR → RGB 변환
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 주의:
    # cv2.imwrite()는 BGR 기준으로 저장하는 것이 자연스럽습니다.
    # RGB 이미지를 그대로 cv2.imwrite()로 저장하면 색상이 뒤틀려 보일 수 있습니다.
    # 따라서 비교용으로만 사용하고, 저장할 때는 다시 BGR로 변환합니다.
    rgb_saved_as_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    rgb_path = os.path.join(output_dir, "03_rgb_converted_back_to_bgr.png")
    save_image(rgb_path, rgb_saved_as_bgr)
    print(f"RGB 변환 후 다시 BGR로 저장: {rgb_path}")

    # 3. BGR → HSV 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV 이미지는 사람이 바로 보기 어렵습니다.
    # 실무에서는 HSV 자체보다 마스크 결과를 주로 확인합니다.
    print("BGR 이미지를 HSV 이미지로 변환 완료")

    # 4. 초록색 영역 검출
    # OpenCV HSV 범위:
    # H: 0~179, S: 0~255, V: 0~255
    lower_green = np.array([40, 80, 80])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    green_mask_path = os.path.join(output_dir, "03_green_mask.png")
    save_image(green_mask_path, green_mask)
    print(f"초록색 마스크 저장: {green_mask_path}")

    green_result = cv2.bitwise_and(image, image, mask=green_mask)

    green_result_path = os.path.join(output_dir, "03_green_result.png")
    save_image(green_result_path, green_result)
    print(f"초록색 검출 결과 저장: {green_result_path}")

    # 5. 파란색 영역 검출
    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([130, 255, 255])

    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    blue_mask_path = os.path.join(output_dir, "03_blue_mask.png")
    save_image(blue_mask_path, blue_mask)
    print(f"파란색 마스크 저장: {blue_mask_path}")

    blue_result = cv2.bitwise_and(image, image, mask=blue_mask)

    blue_result_path = os.path.join(output_dir, "03_blue_result.png")
    save_image(blue_result_path, blue_result)
    print(f"파란색 검출 결과 저장: {blue_result_path}")

    # 6. 빨간색 영역 검출
    # 빨강은 Hue 범위가 0 근처와 179 근처로 나뉩니다.
    lower_red_1 = np.array([0, 80, 80])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 80, 80])
    upper_red_2 = np.array([179, 255, 255])

    red_mask_1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    red_mask_2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)

    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    red_mask_path = os.path.join(output_dir, "03_red_mask.png")
    save_image(red_mask_path, red_mask)
    print(f"빨간색 마스크 저장: {red_mask_path}")

    red_result = cv2.bitwise_and(image, image, mask=red_mask)

    red_result_path = os.path.join(output_dir, "03_red_result.png")
    save_image(red_result_path, red_result)
    print(f"빨간색 검출 결과 저장: {red_result_path}")

    print("=== 색상 공간 변환 및 색상 검출 실습 완료 ===")


if __name__ == "__main__":
    main()