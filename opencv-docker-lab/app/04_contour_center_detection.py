# app/04_contour_center_detection.py

"""
OpenCV 윤곽선 검출과 색상 객체 중심 좌표 계산 실습

실습 내용:
1. 색상 객체가 포함된 테스트 이미지 생성
2. BGR 이미지를 HSV로 변환
3. 특정 색상 마스크 생성
4. cv2.findContours()로 윤곽선 검출
5. 작은 노이즈 제거
6. 가장 큰 객체 선택
7. 바운딩 박스와 중심 좌표 계산
8. 결과 이미지 저장

주의:
- OpenCV 기본 색상 순서는 BGR입니다.
- HSV의 H 범위는 0~179입니다.
- cv2.imshow()는 사용하지 않고 outputs 폴더에 저장합니다.
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


def create_test_image() -> np.ndarray:
    """
    실습용 테스트 이미지를 생성합니다.
    실제 카메라 영상 대신 코드에서 직접 이미지를 만들어 사용합니다.
    """

    height = 480
    width = 640

    # 흰색 배경 생성
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    # OpenCV는 BGR 순서입니다.
    red_bgr = (0, 0, 255)
    green_bgr = (0, 255, 0)
    blue_bgr = (255, 0, 0)
    black_bgr = (0, 0, 0)

    # 검출 대상: 초록색 큰 원
    cv2.circle(
        image,
        center=(420, 260),
        radius=70,
        color=green_bgr,
        thickness=-1,
    )

    # 검출 대상이 아닌 빨간 사각형
    cv2.rectangle(
        image,
        pt1=(70, 80),
        pt2=(200, 200),
        color=red_bgr,
        thickness=-1,
    )

    # 검출 대상이 아닌 파란 원
    cv2.circle(
        image,
        center=(180, 360),
        radius=55,
        color=blue_bgr,
        thickness=-1,
    )

    # 작은 초록색 노이즈들
    cv2.circle(image, center=(80, 420), radius=8, color=green_bgr, thickness=-1)
    cv2.circle(image, center=(580, 70), radius=6, color=green_bgr, thickness=-1)
    cv2.circle(image, center=(550, 390), radius=10, color=green_bgr, thickness=-1)

    # 화면 중앙 기준선 표시
    cv2.line(image, (width // 2, 0), (width // 2, height), black_bgr, 1)
    cv2.line(image, (0, height // 2), (width, height // 2), black_bgr, 1)

    cv2.putText(
        image,
        "Target: GREEN Object",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        black_bgr,
        2,
    )

    return image


def detect_green_object(image: np.ndarray):
    """
    초록색 객체를 검출하고 가장 큰 객체의 정보를 반환합니다.
    """

    # 1. BGR → HSV 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. 초록색 HSV 범위 설정
    lower_green = np.array([40, 80, 80])
    upper_green = np.array([85, 255, 255])

    # 3. 초록색 마스크 생성
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 4. 노이즈 완화를 위한 Morphology 연산
    # 작은 점을 줄이고 객체 영역을 더 안정적으로 만듭니다.
    kernel = np.ones((5, 5), np.uint8)

    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # 5. 윤곽선 찾기
    contours, _ = cv2.findContours(
        mask_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return mask, mask_clean, None

    # 6. 작은 노이즈 제거
    min_area = 1000
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= min_area:
            valid_contours.append(contour)

    if not valid_contours:
        return mask, mask_clean, None

    # 7. 가장 큰 객체 선택
    largest_contour = max(valid_contours, key=cv2.contourArea)

    # 8. 바운딩 박스 계산
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 9. 중심 좌표 계산
    center_x = x + w // 2
    center_y = y + h // 2

    area = cv2.contourArea(largest_contour)

    object_info = {
        "contour": largest_contour,
        "area": area,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "center_x": center_x,
        "center_y": center_y,
    }

    return mask, mask_clean, object_info


def draw_detection_result(image: np.ndarray, object_info):
    """
    검출 결과를 이미지 위에 시각화합니다.
    """

    result = image.copy()

    height, width = result.shape[:2]
    screen_center_x = width // 2
    screen_center_y = height // 2

    if object_info is None:
        cv2.putText(
            result,
            "No green object detected",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return result

    x = object_info["x"]
    y = object_info["y"]
    w = object_info["w"]
    h = object_info["h"]
    center_x = object_info["center_x"]
    center_y = object_info["center_y"]
    area = object_info["area"]

    # 바운딩 박스 그리기
    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (0, 0, 0),
        3,
    )

    # 객체 중심점 그리기
    cv2.circle(
        result,
        (center_x, center_y),
        8,
        (0, 0, 255),
        -1,
    )

    # 화면 중심에서 객체 중심까지 선 그리기
    cv2.line(
        result,
        (screen_center_x, screen_center_y),
        (center_x, center_y),
        (255, 0, 255),
        2,
    )

    # 중심 오차 계산
    error_x = center_x - screen_center_x
    error_y = center_y - screen_center_y

    # 정보 텍스트 출력
    info_text_1 = f"center=({center_x}, {center_y}) area={area:.1f}"
    info_text_2 = f"error_x={error_x}, error_y={error_y}"

    cv2.putText(
        result,
        info_text_1,
        (30, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )

    cv2.putText(
        result,
        info_text_2,
        (30, 455),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )

    # 로봇 제어 관점의 간단한 판단
    if error_x < -50:
        command = "Object is LEFT -> turn left"
    elif error_x > 50:
        command = "Object is RIGHT -> turn right"
    else:
        command = "Object is CENTER -> go forward"

    cv2.putText(
        result,
        command,
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )

    return result


def main():
    print("=== 윤곽선 검출 및 중심 좌표 계산 실습 시작 ===")

    output_dir = "/workspace/outputs"
    ensure_dir(output_dir)

    # 1. 테스트 이미지 생성
    image = create_test_image()

    original_path = os.path.join(output_dir, "04_original_scene.png")
    save_image(original_path, image)
    print(f"원본 테스트 이미지 저장: {original_path}")

    # 2. 초록색 객체 검출
    mask, mask_clean, object_info = detect_green_object(image)

    mask_path = os.path.join(output_dir, "04_green_mask_raw.png")
    save_image(mask_path, mask)
    print(f"초록색 원본 마스크 저장: {mask_path}")

    mask_clean_path = os.path.join(output_dir, "04_green_mask_clean.png")
    save_image(mask_clean_path, mask_clean)
    print(f"초록색 정제 마스크 저장: {mask_clean_path}")

    # 3. 검출 결과 시각화
    result = draw_detection_result(image, object_info)

    result_path = os.path.join(output_dir, "04_detection_result.png")
    save_image(result_path, result)
    print(f"검출 결과 이미지 저장: {result_path}")

    # 4. 콘솔에 객체 정보 출력
    if object_info is None:
        print("초록색 객체를 찾지 못했습니다.")
    else:
        print("초록색 객체 검출 성공")
        print(f"면적: {object_info['area']:.1f}")
        print(f"바운딩 박스: x={object_info['x']}, y={object_info['y']}, w={object_info['w']}, h={object_info['h']}")
        print(f"중심 좌표: center_x={object_info['center_x']}, center_y={object_info['center_y']}")

        height, width = image.shape[:2]
        screen_center_x = width // 2
        error_x = object_info["center_x"] - screen_center_x

        print(f"화면 중심 기준 x 오차: {error_x}")

        if error_x < -50:
            print("판단: 객체가 화면 왼쪽에 있습니다. 로봇은 왼쪽으로 회전해야 합니다.")
        elif error_x > 50:
            print("판단: 객체가 화면 오른쪽에 있습니다. 로봇은 오른쪽으로 회전해야 합니다.")
        else:
            print("판단: 객체가 화면 중앙에 있습니다. 로봇은 전진 또는 집기 동작을 수행할 수 있습니다.")

    print("=== 윤곽선 검출 및 중심 좌표 계산 실습 완료 ===")


if __name__ == "__main__":
    main()