# CUDA-ready adaptamem. OpenMM CUDA platform is present before the container starts.
# Pin: runpod/pytorch 1.0.2, CUDA 12.8.1, Ubuntu 24.04.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace/adaptamem

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir 'openmm[cuda12]' pdbfixer mdtraj pyyaml

COPY . .
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
CMD ["bash", "scripts/runpod_boot.sh"]
