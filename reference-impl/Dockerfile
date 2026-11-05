# Reference Implementation Runtime: Paged KV-Cache Transformer Server + Live K8s AIOps Controller
FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

WORKDIR /app

COPY reference-impl/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    groupadd -g 10001 aiops && \
    useradd -u 10001 -g 10001 -M -s /usr/sbin/nologin aiops && \
    chown -R 10001:10001 /app

COPY aiops /app/aiops
COPY reference-impl/app /app/ref_app

USER 10001:10001
EXPOSE 8000

ENTRYPOINT ["python3", "-m", "ref_app.server"]
