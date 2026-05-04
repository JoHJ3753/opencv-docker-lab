# app/05_canny_edge_detection.py

"""
OpenCV 이미지 필터링과 Canny Edge Detection 실습

실습 내용:
1. 테스트 이미지 생성
2. 흑백 변환
3. Gaussian Blur 적용
4. Canny Edge Detection 적용
5. threshold 값 변화 비교
6. 결과 이미지 저장

주의:
- cv2.imshow()는 사용하지 않고 outputs 폴더에 저장합니다.
- Docker 컨테이너 내부 기준 출력 경로는 /workspace/outputs 입니다.
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


def create_edge_test_image() -> np.ndarray:
    """
    에지 검출 실습용 테스트 이미지를 생성합니다.
    실제 카메라 이미지 대신 도형과 선이 포함된 이미지를 만듭니다.
    """

    height = 480
    width = 640

    # 흰색 배경
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    black = (0, 0, 0)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    green = (0, 255, 0)
    gray = (180, 180, 180)

    # 큰 사각형
    cv2.rectangle(image, (60, 80), (240, 240), black, 3)

    # 채워진 원
    cv2.circle(image, (430, 160), 80, blue, -1)

    # 삼각형
    points = np.array([[320, 330], [460, 430], [200, 430]], np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=red, thickness=4)

    # 로봇 주행 라인처럼 보이는 두 선
    cv2.line(image, (90, 460), (280, 280), green, 5)
    cv2.line(image, (550, 460), (360, 280), green, 5)

    # 약한 회색 선
    cv2.line(image, (20, 40), (620, 40), gray, 2)

    # 작은 노이즈 점 생성
    rng = np.random.default_rng(seed=42)

    for _ in range(200):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        color_value = int(rng.integers(0, 255))
        image[y, x] = (color_value, color_value, color_value)

    cv2.putText(
        image,
        "Canny Edge Test",
        (190, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        black,
        2,
    )

    return image


def main():
    print("=== Canny Edge Detection 실습 시작 ===")

    output_dir = "/workspace/outputs"
    ensure_dir(output_dir)

    # 1. 테스트 이미지 생성
    image = create_edge_test_image()

    original_path = os.path.join(output_dir, "05_original_edge_test.png")
    save_image(original_path, image)
    print(f"원본 이미지 저장: {original_path}")

    # 2. 흑백 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_path = os.path.join(output_dir, "05_gray.png")
    save_image(gray_path, gray)
    print(f"흑백 이미지 저장: {gray_path}")

    # 3. Gaussian Blur 적용
    # 커널 크기는 홀수여야 합니다. 예: 3, 5, 7
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    blurred_path = os.path.join(output_dir, "05_blurred.png")
    save_image(blurred_path, blurred)
    print(f"Gaussian Blur 이미지 저장: {blurred_path}")

    # 4. Canny Edge Detection 적용
    edges_50_150 = cv2.Canny(blurred, 50, 150)

    edges_50_150_path = os.path.join(output_dir, "05_canny_50_150.png")
    save_image(edges_50_150_path, edges_50_150)
    print(f"Canny 결과 저장 threshold 50/150: {edges_50_150_path}")

    # 5. threshold 값 변화 비교
    edges_30_100 = cv2.Canny(blurred, 30, 100)
    edges_100_200 = cv2.Canny(blurred, 100, 200)
    edges_150_250 = cv2.Canny(blurred, 150, 250)

    save_image(os.path.join(output_dir, "05_canny_30_100_more_sensitive.png"), edges_30_100)
    save_image(os.path.join(output_dir, "05_canny_100_200_standard.png"), edges_100_200)
    save_image(os.path.join(output_dir, "05_canny_150_250_less_sensitive.png"), edges_150_250)

    print("threshold 비교 결과 저장 완료")

    # 6. Blur 없이 Canny 적용해서 비교
    edges_without_blur = cv2.Canny(gray, 50, 150)

    edges_without_blur_path = os.path.join(output_dir, "05_canny_without_blur.png")
    save_image(edges_without_blur_path, edges_without_blur)
    print(f"Blur 없이 Canny 적용 결과 저장: {edges_without_blur_path}")

    # 7. 컬러 원본 위에 에지 표시
    # edges는 흑백 이미지이므로 컬러 마스크처럼 활용합니다.
    edge_overlay = image.copy()
    edge_overlay[edges_50_150 > 0] = (0, 0, 255)

    overlay_path = os.path.join(output_dir, "05_edge_overlay_red.png")
    save_image(overlay_path, edge_overlay)
    print(f"에지 오버레이 이미지 저장: {overlay_path}")

    print("=== Canny Edge Detection 실습 완료 ===")


if __name__ == "__main__":
    main()