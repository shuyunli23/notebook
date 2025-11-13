#!/usr/bin/env python
# vllm_style_jetson_qwen3vl.py
import asyncio
import json
import logging
import time
import gc
import os
import threading
from typing import Dict, Any, List, Optional, Tuple, Union
from queue import Queue, Empty, PriorityQueue
from dataclasses import dataclass
import hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import click
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor

# vLLM 风格的环境优化
os.environ.update({
    "CUDA_LAUNCH_BLOCKING": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.8",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    "NCCL_NVLS_ENABLE": "0",
    "CUDA_MODULE_LOADING": "LAZY",
    "TORCH_CUDNN_SDPA_ENABLED": "1"
})

# 极致性能设置
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')  # 使用最高精度的 TF32

# 优化线程设置
torch.set_num_threads(8)
torch.set_num_interop_threads(4)

@dataclass
class GenerationRequest:
    """生成请求"""
    request_id: str
    messages: List[Dict]
    max_tokens: int
    temperature: float
    priority: int
    created_time: float
    future: asyncio.Future
    
    def __lt__(self, other):
        return self.priority < other.priority

class VLLMStyleKVCache:
    """vLLM 风格的 KV 缓存管理"""
    
    def __init__(self, max_blocks: int = 1000, block_size: int = 16):
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.free_blocks = list(range(max_blocks))
        self.allocated_blocks = {}
        self.block_tables = {}
        self.lock = threading.Lock()
    
    def allocate_blocks(self, seq_id: str, num_blocks: int) -> List[int]:
        """为序列分配块"""
        with self.lock:
            if len(self.free_blocks) < num_blocks:
                # 回收最老的块
                self._evict_oldest_blocks(num_blocks - len(self.free_blocks))
            
            allocated = []
            for _ in range(min(num_blocks, len(self.free_blocks))):
                block_id = self.free_blocks.pop(0)
                allocated.append(block_id)
            
            self.allocated_blocks[seq_id] = allocated
            self.block_tables[seq_id] = allocated.copy()
            return allocated
    
    def free_blocks(self, seq_id: str):
        """释放序列的块"""
        with self.lock:
            if seq_id in self.allocated_blocks:
                blocks = self.allocated_blocks.pop(seq_id)
                self.free_blocks.extend(blocks)
                self.block_tables.pop(seq_id, None)
    
    def _evict_oldest_blocks(self, num_blocks: int):
        """回收最老的块"""
        # 简单的 LRU 策略
        if len(self.allocated_blocks) > 0:
            oldest_seq = list(self.allocated_blocks.keys())[0]
            self.free_blocks(oldest_seq)

class ContinuousBatchProcessor:
    """连续批处理器 - vLLM 风格"""
    
    def __init__(self, model, processor, max_batch_size: int = 8):
        self.model = model
        self.processor = processor
        self.max_batch_size = max_batch_size
        self.device = next(model.parameters()).device
        
        # 请求队列
        self.pending_requests = PriorityQueue()
        self.running_requests = {}
        self.completed_requests = {}
        
        # KV 缓存管理
        self.kv_cache = VLLMStyleKVCache()
        
        # 批处理状态
        self.current_batch = []
        self.batch_lock = threading.Lock()
        
        # 启动连续批处理循环
        self.processing_thread = threading.Thread(target=self._continuous_batching_loop, daemon=True)
        self.processing_thread.start()
        
        # 预编译的生成配置
        self.generation_configs = self._prepare_optimized_configs()
    
    def _prepare_optimized_configs(self):
        """预准备优化的生成配置"""
        configs = {}
        
        # 不同场景的优化配置
        configs['greedy'] = {
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "repetition_penalty": 1.02,
            "length_penalty": 1.0,
            "early_stopping": True,
        }
        
        configs['sampling'] = {
            "do_sample": True,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "repetition_penalty": 1.02,
            "length_penalty": 1.0,
            "early_stopping": True,
            "top_p": 0.9,
            "top_k": 50,
        }
        
        return configs
    
    async def add_request(self, request: GenerationRequest) -> Dict[str, Any]:
        """添加请求到队列"""
        # 添加到待处理队列
        await asyncio.get_event_loop().run_in_executor(
            None, self.pending_requests.put, request
        )
        
        # 等待结果
        try:
            result = await asyncio.wait_for(request.future, timeout=300)
            return result
        except asyncio.TimeoutError:
            # 清理超时请求
            self.kv_cache.free_blocks(request.request_id)
            raise Exception("Request timeout")
    
    def _continuous_batching_loop(self):
        """连续批处理主循环"""
        while True:
            try:
                # 收集待处理请求
                self._collect_pending_requests()
                
                # 如果有请求需要处理
                if self.current_batch:
                    self._process_current_batch()
                else:
                    time.sleep(0.001)  # 短暂休眠
                    
            except Exception as e:
                logging.error(f"Continuous batching error: {e}")
                time.sleep(0.01)
    
    def _collect_pending_requests(self):
        """收集待处理的请求"""
        with self.batch_lock:
            # 移除已完成的请求
            self.current_batch = [req for req in self.current_batch 
                                if not req.future.done()]
            
            # 添加新请求到当前批次
            while (len(self.current_batch) < self.max_batch_size and 
                   not self.pending_requests.empty()):
                try:
                    request = self.pending_requests.get_nowait()
                    self.current_batch.append(request)
                    self.running_requests[request.request_id] = request
                except Empty:
                    break
    
    def _process_current_batch(self):
        """处理当前批次"""
        if not self.current_batch:
            return
        
        try:
            # 按请求类型分组处理
            text_only_requests = []
            vision_requests = []
            
            for request in self.current_batch:
                text_prompt, image_urls = self._extract_content(request.messages)
                if image_urls:
                    vision_requests.append(request)
                else:
                    text_only_requests.append(request)
            
            # 优先处理纯文本请求（更快）
            if text_only_requests:
                self._process_text_batch(text_only_requests)
            
            # 处理视觉请求
            if vision_requests:
                self._process_vision_batch(vision_requests)
                
        except Exception as e:
            logging.error(f"Batch processing error: {e}")
            # 标记所有请求为失败
            for request in self.current_batch:
                if not request.future.done():
                    request.future.set_exception(e)
    
    def _process_vision_batch(self, requests: List[GenerationRequest]):
        """处理视觉请求批次"""
        if not requests:
            return
        
        # 为了简化，逐个处理视觉请求（视觉模型批处理较复杂）
        for request in requests:
            try:
                result = self._process_single_vision_request(request)
                if not request.future.done():
                    request.future.set_result(result)
            except Exception as e:
                if not request.future.done():
                    request.future.set_exception(e)
    
    def _process_single_vision_request(self, request: GenerationRequest) -> Dict[str, Any]:
        """处理单个视觉请求 - 极致优化"""
        start_time = time.time()
        
        # 提取内容
        text_prompt, image_urls = self._extract_content(request.messages)
        
        # 构建 Qwen 格式消息
        content_list = []
        for img_url in image_urls:
            content_list.append({"type": "image", "image": img_url})
        content_list.append({"type": "text", "text": text_prompt})
        qwen_messages = [{"role": "user", "content": content_list}]
        
        # 准备输入 - 使用流水线优化
        inputs = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device, non_blocking=True)
        
        # 选择优化的生成配置
        if request.temperature <= 0.01:
            generation_kwargs = self.generation_configs['greedy'].copy()
        else:
            generation_kwargs = self.generation_configs['sampling'].copy()
            generation_kwargs['temperature'] = max(request.temperature, 0.1)
        
        generation_kwargs['max_new_tokens'] = min(request.max_tokens, 512)
        
        # 使用优化的生成策略
        with torch.no_grad():
            # 启用所有可能的优化
            with torch.cuda.amp.autocast(enabled=True):
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):  # 使用 Flash Attention
                    generated_ids = self.model.generate(**inputs, **generation_kwargs)
        
        # 快速解码
        input_length = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[0][input_length:]
        
        output_text = self.processor.tokenizer.decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        # 立即清理内存
        del inputs, generated_ids
        torch.cuda.empty_cache()
        
        processing_time = time.time() - start_time
        
        return {
            "text": output_text,
            "prompt_tokens": input_length,
            "completion_tokens": len(generated_ids_trimmed),
            "total_tokens": input_length + len(generated_ids_trimmed),
            "processing_time": processing_time,
            "cache_hit": False
        }
    
    def _process_text_batch(self, requests: List[GenerationRequest]):
        """处理纯文本请求批次（可以真正批处理）"""
        # 这里可以实现真正的批处理
        # 为了简化，暂时逐个处理
        for request in requests:
            try:
                result = self._process_single_vision_request(request)
                if not request.future.done():
                    request.future.set_result(result)
            except Exception as e:
                if not request.future.done():
                    request.future.set_exception(e)
    
    def _extract_content(self, messages: List[Dict]) -> Tuple[str, List[str]]:
        """提取消息内容"""
        text_parts = []
        image_urls = []
        
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", [])
                
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            image_url_data = item.get("image_url", {})
                            if isinstance(image_url_data, dict):
                                image_urls.append(image_url_data.get("url", ""))
                            else:
                                image_urls.append(str(image_url_data))
                        elif item.get("type") == "image":
                            image_urls.append(item.get("image", ""))
        
        return " ".join(text_parts).strip(), image_urls

class VLLMStyleJetsonEngine:
    """vLLM 风格的 Jetson 引擎"""
    
    def __init__(self, model_dir: str, dtype=torch.float16):
        self.model_dir = model_dir
        self.dtype = dtype
        
        logging.info(f"🚀 Loading vLLM-Style Qwen3-VL for Jetson Orin")
        
        # GPU 内存优化
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory = gpu_props.total_memory / 1024**3
            logging.info(f"📊 Jetson GPU: {gpu_props.name}, Memory: {gpu_memory:.1f} GB")
            
            # 激进的内存设置
            torch.cuda.set_per_process_memory_fraction(0.9)
            torch.cuda.empty_cache()
        
        # 模型加载 - vLLM 风格优化
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": "cuda:0",
            "use_safetensors": True,
            "attn_implementation": "flash_attention_2",  # 尝试使用 Flash Attention
        }
        
        try:
            logging.info("📦 Loading model with Flash Attention...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_dir, **model_kwargs
            )
        except Exception as e:
            logging.warning(f"Flash Attention failed: {e}")
            # 回退到标准注意力
            model_kwargs.pop("attn_implementation", None)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_dir, **model_kwargs
            )
        
        logging.info("✅ Model loaded successfully")
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.device = self.model.device
        
        # 应用 vLLM 风格优化
        self._apply_vllm_optimizations()
        
        # 创建连续批处理器
        self.batch_processor = ContinuousBatchProcessor(
            self.model, self.processor, max_batch_size=4
        )
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_generated": 0,
            "total_latency": 0.0,
        }
        
        # 预热
        self._vllm_warmup()
        
        logging.info(f"✅ vLLM-Style engine ready!")

    def _apply_vllm_optimizations(self):
        """应用 vLLM 风格优化"""
        try:
            logging.info("🔥 Applying vLLM-style optimizations...")
            
            # 1. 模型编译 - 最激进模式
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(
                        self.model,
                        mode="max-autotune-no-cudagraphs",  # vLLM 风格编译
                        fullgraph=False,
                        dynamic=True  # 支持动态形状
                    )
                    logging.info("✅ Model compiled with vLLM-style settings")
                except Exception as e:
                    logging.warning(f"vLLM-style compilation failed: {e}")
            
            # 2. 设置评估模式
            self.model.eval()
            
            # 3. 冻结参数
            for param in self.model.parameters():
                param.requires_grad = False
            
            # 4. 启用所有缓存优化
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = True
            
            # 5. 启用融合操作
            torch.backends.cuda.enable_flash_sdp(True)
            
            logging.info("✅ vLLM-style optimizations applied")
            
        except Exception as e:
            logging.warning(f"Some vLLM optimizations failed: {e}")

    def _vllm_warmup(self):
        """vLLM 风格预热"""
        try:
            logging.info("🔥 vLLM-style warmup starting...")
            
            # 预热不同长度的序列
            warmup_lengths = [32, 64, 128, 256]
            
            for length in warmup_lengths:
                text = "Hello " * (length // 6)
                messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
                
                try:
                    inputs = self.processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=True):
                            _ = self.model.generate(
                                **inputs,
                                max_new_tokens=10,
                                do_sample=False,
                                use_cache=True,
                                pad_token_id=self.processor.tokenizer.eos_token_id,
                                num_beams=1
                            )
                    
                    del inputs
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logging.warning(f"Warmup failed for length {length}: {e}")
            
            logging.info("✅ vLLM-style warmup completed")
            
        except Exception as e:
            logging.warning(f"vLLM warmup failed: {e}")

    async def generate_async(self, messages: List[Dict], max_tokens: int = 128, 
                           temperature: float = 0.7) -> Dict[str, Any]:
        """异步生成 - vLLM 风格"""
        request_id = f"req_{int(time.time() * 1000000)}"
        future = asyncio.Future()
        
        request = GenerationRequest(
            request_id=request_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=1,  # 可以根据需要调整优先级
            created_time=time.time(),
            future=future
        )
        
        try:
            result = await self.batch_processor.add_request(request)
            
            # 更新统计
            self.stats["successful_requests"] += 1
            self.stats["total_tokens_generated"] += result["completion_tokens"]
            self.stats["total_latency"] += result["processing_time"]
            
            return result
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            raise e
        finally:
            self.stats["total_requests"] += 1

    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_latency = 0.0
        if self.stats["successful_requests"] > 0:
            avg_latency = self.stats["total_latency"] / self.stats["successful_requests"]
        
        gpu_memory_info = {}
        if torch.cuda.is_available():
            gpu_memory_info = {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
        
        return {
            "engine_info": {
                "model_dir": self.model_dir,
                "device": str(self.device),
                "precision": "FP16",
                "platform": "vLLM-Style Jetson Orin"
            },
            "runtime_stats": {
                "total_requests": self.stats["total_requests"],
                "successful_requests": self.stats["successful_requests"],
                "failed_requests": self.stats["failed_requests"],
                "total_tokens_generated": self.stats["total_tokens_generated"],
                "average_latency_ms": avg_latency * 1000,
                "tokens_per_second": self.stats["total_tokens_generated"] / max(self.stats["total_latency"], 0.001)
            },
            "memory_info": gpu_memory_info
        }

class VLLMStyleServer:
    """vLLM 风格服务器"""
    
    def __init__(self, engine: VLLMStyleJetsonEngine):
        self.engine = engine
        self.app = FastAPI(
            title="vLLM-Style Jetson Qwen3-VL Server", 
            version="1.0.0",
            docs_url=None,
            redoc_url=None
        )
        self.register_routes()

    def register_routes(self):
        self.app.add_api_route("/stats", self.stats, methods=["GET"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/v1/chat/completions", self.chat_completions, methods=["POST"])

    async def stats(self) -> Response:
        stats = await self.engine.get_stats()
        return JSONResponse(stats)

    async def health(self) -> Response:
        return Response(status_code=200)

    async def chat_completions(self, request: Request) -> Response:
        try:
            request_dict = await request.json()
            messages = request_dict.get("messages", [])
            max_tokens = min(request_dict.get("max_tokens", 128), 512)
            temperature = request_dict.get("temperature", 0.7)
            
            result = await self.engine.generate_async(messages, max_tokens, temperature)
            
            response_data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "Qwen3-VL-4B-Instruct-vLLM-Style",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["text"]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"]
                },
                "processing_time": result["processing_time"]
            }
            
            return JSONResponse(response_data)
            
        except Exception as e:
            logging.error(f"Request failed: {e}")
            return JSONResponse({
                "error": {"message": str(e), "type": "internal_server_error"}
            }, status_code=500)

    async def __call__(self, host: str, port: int):
        config = uvicorn.Config(
            self.app, 
            host=host, 
            port=port, 
            log_level="warning",
            access_log=False,
            workers=1
        )
        await uvicorn.Server(config).serve()

@click.command()
@click.argument("model_dir")
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--port", type=int, default=8001)
@click.option("--dtype", type=click.Choice(["float16", "bfloat16"]), default="float16")
def entrypoint(model_dir: str, host: str, port: int, dtype: str):
    """
    启动 vLLM 风格的 Jetson Orin Qwen3-VL API 服务器
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]
    
    logging.info(f"🚀 Starting vLLM-Style Jetson Qwen3-VL Server")
    logging.info(f"📍 Model: {model_dir}")
    logging.info(f"🌐 Server: http://{host}:{port}")
    logging.info(f"💾 Precision: {dtype}")
    
    # 创建引擎
    engine = VLLMStyleJetsonEngine(
        model_dir=model_dir,
        dtype=torch_dtype
    )
    
    # 创建服务器
    server = VLLMStyleServer(engine)
    
    # 启动服务器
    asyncio.run(server(host, port))

if __name__ == "__main__":
    entrypoint()
