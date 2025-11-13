vllm serve /home/quan/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct \
  --max-model-len 16384 \
  --dtype float16 \
  --host 0.0.0.0 \
  --port 8001


vllm serve /home/quan/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct \
  --max-model-len 4096 \
  --dtype float16 \
  --host 0.0.0.0 \
  --port 8001

vllm serve /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct \
  --max-model-len 16384 \
  --dtype float16 \
  --host 0.0.0.0 \
  --port 8001

vllm serve /home/quan/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct \
  --max-model-len 16384 \
  --dtype float16 \
  --host 0.0.0.0 \
  --port 8001