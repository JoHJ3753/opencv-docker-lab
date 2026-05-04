# app/06_roi_line_detection.py

"""
OpenCV ROI 관심 영역 설정과 라인 검출 기초 실습

실습 내용:
1. 주행선이 포함된 테스트 이미지 생성
2. Gray 변환
3. Gaussian Blur
4. Canny Edge Detection
5. 사다리꼴 ROI 마스크 생성
6. ROI 내부 에지만 추출
7. HoughLinesP로 직선 검출
8. 검출된 선을 원본 이미지에 표시

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


def create_lane_test_image() -> np.ndarray:
    """
    간단한 주행선 테스트 이미지를 생성합니다.
    실제 카메라 이미지 대신 코드로 직접 차선/라인이 있는 장면을 만듭니다.
    """

    height = 480
    width = 640

    # 어두운 회색 배경: 바닥 느낌
    image = np.full((height, width, 3), 70, dtype=np.uint8)

    white = (255, 255, 255)
    yellow = (0, 255, 255)
    gray = (130, 130, 130)
    red = (0, 0, 255)
    black = (0, 0, 0)

    # 원근감 있는 주행선 두 개
    cv2.line(image, (160, height), (290, 270), white, 8)
    cv2.line(image, (500, height), (350, 270), yellow, 8)

    # 바닥 노이즈성 선들
    cv2.line(image, (40, 120), (250, 160), gray, 2)
    cv2.line(image, (390, 130), (620, 180), gray, 2)
    cv2.rectangle(image, (40, 300), (120, 360), gray, 2)

    # 화면 상단에 불필요한 선 추가
    cv2.line(image, (20, 50), (620, 50), red, 4)
    cv2.line(image, (80, 90), (560, 110), red, 3)

    cv2.putText(
        image,
        "ROI + Line Detection",
        (170, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        white,
        2,
    )

    # 화면 중심선
    cv2.line(image, (width // 2, 0), (width // 2, height), black, 1)

    return image


def apply_roi_mask(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    에지 이미지에 사다리꼴 ROI 마스크를 적용합니다.

    반환:
    - roi_edges: ROI 영역만 남긴 에지 이미지
    - roi_mask: ROI 영역이 흰색인 마스크 이미지
    """

    height, width = edges.shape

    # 검은 마스크 생성
    mask = np.zeros_like(edges)

    # 사다리꼴 ROI 좌표 설정
    # 좌표 순서: 왼쪽 아래, 왼쪽 위, 오른쪽 위, 오른쪽 아래
    polygon = np.array(
        [
            [
                (80, height),
                (260, int(height * 0.55)),
                (380, int(height * 0.55)),
                (580, height),
            ]
        ],
        dtype=np.int32,
    )

    # ROI 영역을 흰색으로 채움
    cv2.fillPoly(mask, polygon, 255)

    # 에지 이미지와 마스크를 AND 연산
    roi_edges = cv2.bitwise_and(edges, mask)

    return roi_edges, mask


def detect_lines(roi_edges: np.ndarray):
    """
    ROI 내부 에지 이미지에서 직선을 검출합니다.
    """

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=40,
        maxLineGap=35,
    )

    return lines


def draw_lines(image: np.ndarray, lines) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    검출된 선을 이미지 위에 표시합니다.
    """

    result = image.copy()
    detected_lines = []

    if lines is None:
        cv2.putText(
            result,
            "No lines detected",
            (30, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return result, detected_lines

    for line in lines:
        x1, y1, x2, y2 = line[0]
        detected_lines.append((x1, y1, x2, y2))

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            4,
        )

    cv2.putText(
        result,
        f"Detected lines: {len(detected_lines)}",
        (30, 440),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return result, detected_lines


def draw_roi_polygon(image: np.ndarray) -> np.ndarray:
    """
    원본 이미지 위에 ROI 영역을 표시합니다.
    """

    result = image.copy()
    height, width = result.shape[:2]

    polygon = np.array(
        [
            [
                (80, height),
                (260, int(height * 0.55)),
                (380, int(height * 0.55)),
                (580, height),
            ]
        ],
        dtype=np.int32,
    )

    cv2.polylines(
        result,
        polygon,
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
    )

    return result


def analyze_lane_direction(detected_lines, image_width: int) -> str:
    """
    검출된 선들의 대략적인 위치를 이용해 간단한 주행 판단을 수행합니다.

    실제 자율주행 알고리즘은 훨씬 복잡하지만,
    초급 과정에서는 선들의 평균 x 위치로 개념만 이해합니다.
    """

    if not detected_lines:
        return "라인 미검출: 정지 또는 저속 탐색 필요"

    x_centers = []

    for x1, y1, x2, y2 in detected_lines:
        x_centers.append((x1 + x2) / 2)

    avg_x = sum(x_centers) / len(x_centers)
    screen_center_x = image_width / 2
    error_x = avg_x - screen_center_x

    if error_x < -40:
        return f"라인 중심이 왼쪽: 좌측 보정 필요, error_x={error_x:.1f}"
    elif error_x > 40:
        return f"라인 중심이 오른쪽: 우측 보정 필요, error_x={error_x:.1f}"
    else:
        return f"라인이 중앙 근처: 직진 가능, error_x={error_x:.1f}"


def main():
    print("=== ROI 관심 영역 및 라인 검출 실습 시작 ===")

    output_dir = "/workspace/outputs"
    ensure_dir(output_dir)

    # 1. 테스트 이미지 생성
    image = create_lane_test_image()
    height, width = image.shape[:2]

    original_path = os.path.join(output_dir, "06_original_lane_scene.png")
    save_image(original_path, image)
    print(f"원본 주행선 이미지 저장: {original_path}")

    # 2. ROI 표시 이미지 저장
    roi_visual = draw_roi_polygon(image)
    roi_visual_path = os.path.join(output_dir, "06_roi_visualization.png")
    save_image(roi_visual_path, roi_visual)
    print(f"ROI 시각화 이미지 저장: {roi_visual_path}")

    # 3. Gray 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(output_dir, "06_gray.png")
    save_image(gray_path, gray)
    print(f"Gray 이미지 저장: {gray_path}")

    # 4. Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred_path = os.path.join(output_dir, "06_blurred.png")
    save_image(blurred_path, blurred)
    print(f"Blur 이미지 저장: {blurred_path}")

    # 5. Canny Edge
    edges = cv2.Canny(blurred, 50, 150)
    edges_path = os.path.join(output_dir, "06_edges_full.png")
    save_image(edges_path, edges)
    print(f"전체 에지 이미지 저장: {edges_path}")

    # 6. ROI 마스크 적용
    roi_edges, roi_mask = apply_roi_mask(edges)

    roi_mask_path = os.path.join(output_dir, "06_roi_mask.png")
    save_image(roi_mask_path, roi_mask)
    print(f"ROI 마스크 저장: {roi_mask_path}")

    roi_edges_path = os.path.join(output_dir, "06_edges_roi_only.png")
    save_image(roi_edges_path, roi_edges)
    print(f"ROI 내부 에지 저장: {roi_edges_path}")

    # 7. HoughLinesP로 라인 검출
    lines = detect_lines(roi_edges)

    # 8. 검출 라인 시각화
    result, detected_lines = draw_lines(image, lines)

    result_path = os.path.join(output_dir, "06_line_detection_result.png")
    save_image(result_path, result)
    print(f"라인 검출 결과 저장: {result_path}")

    # 9. 주행 판단
    decision = analyze_lane_direction(detected_lines, width)

    print(f"검출된 라인 수: {len(detected_lines)}")
    print(f"간단 주행 판단: {decision}")

    print("검출 라인 좌표 일부:")
    for idx, line in enumerate(detected_lines[:10], start=1):
        x1, y1, x2, y2 = line
        print(f"  line {idx}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    print("=== ROI 관심 영역 및 라인 검출 실습 완료 ===")


if __name__ == "__main__":
    main()