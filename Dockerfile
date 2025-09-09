FROM runpod/base:0.6.1-cuda12.2.0

# Python dependencies
COPY requirements.txt .
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

COPY rp_handler.py /rp_handler.py
CMD python3.10 -u /rp_handler.py