FROM runpod/base:0.6.1-cuda12.2.0

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install -r requirements.txt --no-cache-dir && \
    rm requirements.txt

COPY rp_handler.py /rp_handler.py

CMD ["python3.10", "-u", "/rp_handler.py"]
