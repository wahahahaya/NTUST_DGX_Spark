# NTUST NVIDIA SGX Spark

# Installation and Environment Setup

This guide describes how to install dependencies, download containers, and run services for TensorRT-LLM (gpt-oss), YOLOv11, and Open-WebUI.

---

# 1. Install Python Packages

```bash
pip install OpenAI open-webui
```

# 2. Download and Run Containers
## 2.1 TensorRT-LLM (gpt-oss)
### 2.1.1 Verify GPU Environment
```
docker run --rm --gpus all nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev nvidia-smi
```
### 2.1.2 Set Model Name
```
export MODEL_HANDLE="openai/gpt-oss-20b"
```
You can replace it with another supported model.
### 2.1.3 Launch TensorRT-LLM Server (Port 8355)
```
docker run --name trtllm_llm_server --rm -it --gpus all --ipc host --network host \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c '
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken && \
    hf download $MODEL_HANDLE && \
    cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.4
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF
    trtllm-serve "$MODEL_HANDLE" \
      --max_batch_size 32 \
      --trust_remote_code \
      --port 8355 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml
  '
```
### 2.1.4 Verify Server Availability
```
curl -X POST "http://localhost:8355/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-oss",
        "messages": [
          {"role": "user", "content": "Hello"}
        ]
      }'
```
## 2.2 YOLOv11
### 2.2.1 Pull the ARM64 Image
```
docker pull ultralytics/ultralytics:latest-arm64
```
### 2.2.2 Run YOLOv11 Container
```
sudo docker run --rm -it --ipc=host --runtime=nvidia --gpus all \
  -p 6611:6611 \
  -v /home/ntust_spark/playbook_gptoss/ultra_runs:/ultralytics/runs \
  -v /home/ntust_spark/playbook_gptoss/yolo_server:/ultralytics/app \
  ultralytics/ultralytics:latest-arm64
```
### 2.2.3 Install Dependencies Inside Container
```
pip install fastapi uvicorn pillow python-multipart
```
### 2.2.4 Launch YOLOv11 Server (Port 6611)
```
cd app
python server.py
```
### 2.2.5 Verify Server Availability
- Health Check
```
curl http://localhost:6611/health
```
Expected output (example):
```
{"status": "ok"}
```
- Inference Example
Assuming the API exposes /predict and accepts image upload:
```
curl -X POST "http://localhost:6611/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.jpg"
```
You should receive detection results in JSON format.
## NVIDIA DeepStream
### 2.3.1 Install the Spark Image
```
docker run --rm --gpus all nvcr.io/nvidia/deepstream:8.0-triton-dgx-spark nvidia-smi
```
## Open-WebUI
### 2.4.1 Start Open-WebUI (Port 9999)
```
DATA_DIR=/home/ntust_spark/playbook_gptoss/webui open-webui serve --port 9999
```
### 2.4.2 Access the UI
Open the browser: http://localhost:9999
