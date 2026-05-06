FROM python:3.11-slim

# 컨테이너 내부 작업 폴더
WORKDIR /workspace

# Python 실행 시 로그가 바로 출력되도록 설정
ENV PYTHONUNBUFFERED=1

# pip 최신화 및 기본 패키지 설치
RUN python -m pip install --upgrade pip

# requirements.txt를 먼저 복사
# 이렇게 하면 requirements.txt가 바뀌지 않았을 때 Docker 캐시를 활용할 수 있음
COPY requirements.txt /workspace/requirements.txt

# Python 패키지 설치
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# 기본 실행 위치
CMD ["bash"]
