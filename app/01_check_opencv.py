# app/01_check_opencv.py

"""
첫 번째 OpenCV 실행 테스트 파일

목표:
1. Python 실행 확인
2. OpenCV(cv2) 설치 확인
3. NumPy 설치 확인
4. 테스트 이미지 생성
5. OpenCV로 이미지 저장/읽기/변환
6. 결과를 outputs 폴더에 저장

주의:
- Docker 컨테이너 내부 기준 경로는 /workspace 입니다.
- app, data, outputs 폴더는 docker-compose.yml에서 볼륨으로 연결되어 있습니다.
"""

import os
import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    """
    폴더가 없으면 생성하는 함수입니다.
    Docker 볼륨 연결 상태에서도 안전하게 사용할 수 있습니다.
    """
    os.makedirs(path, exist_ok=True)


def main():
    print("=== OpenCV Docker Lab 실행 테스트 ===")

    # 1. 버전 확인
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")

    # 2. 컨테이너 내부 기준 경로 설정
    image_dir = "/workspace/data/images"
    output_dir = "/workspace/outputs"

    ensure_dir(image_dir)
    ensure_dir(output_dir)

    # 3. 테스트 이미지 생성
    # 300x500 크기의 검은색 이미지 생성
    height = 300
    width = 500
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 4. 이미지 위에 도형과 글자 그리기
    # OpenCV 색상 순서는 RGB가 아니라 BGR입니다.
    # (255, 0, 0)은 빨강이 아니라 파랑입니다.
    cv2.rectangle(
        image,
        pt1=(50, 50),
        pt2=(450, 250),
        color=(255, 0, 0),
        thickness=3,
    )

    cv2.circle(
        image,
        center=(250, 150),
        radius=60,
        color=(0, 255, 0),
        thickness=-1,
    )

    cv2.putText(
        image,
        text="OpenCV Docker Test",
        org=(80, 285),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(255, 255, 255),
        thickness=2,
    )

    # 5. 원본 테스트 이미지 저장
    input_image_path = os.path.join(image_dir, "opencv_test_input.png")
    cv2.imwrite(input_image_path, image)
    print(f"테스트 이미지 저장 완료: {input_image_path}")

    # 6. OpenCV로 이미지 다시 읽기
    loaded_image = cv2.imread(input_image_path)

    if loaded_image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {input_image_path}")

    print(f"이미지 읽기 성공: shape={loaded_image.shape}")

    # 7. 흑백 변환
    gray_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)

    # 8. 결과 저장
    output_image_path = os.path.join(output_dir, "opencv_test_gray.png")
    cv2.imwrite(output_image_path, gray_image)

    print(f"흑백 변환 결과 저장 완료: {output_image_path}")
    print("=== 테스트 완료 ===")


if __name__ == "__main__":
    main()