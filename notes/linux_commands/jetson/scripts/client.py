# vision_concurrent_client.py
import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Any
import statistics
from datetime import datetime
import argparse
import base64
import requests
from io import BytesIO
from PIL import Image


class VisionConcurrentClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.results = []

    def encode_image_to_base64(self, image_url: str) -> str:
        """将图片URL转换为base64编码"""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            # 转换为base64
            image_base64 = base64.b64encode(response.content).decode('utf-8')

            # 检测图片格式
            content_type = response.headers.get('content-type', 'image/jpeg')
            if 'png' in content_type:
                mime_type = 'image/png'
            elif 'gif' in content_type:
                mime_type = 'image/gif'
            else:
                mime_type = 'image/jpeg'

            return f"data:{mime_type};base64,{image_base64}"

        except Exception as e:
            print(f"❌ 图片加载失败 {image_url}: {e}")
            return None

    async def send_vision_request(self, session: aiohttp.ClientSession, request_id: int,
                                  image_url: str, text_prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """发送视觉理解请求"""

        # 预处理图片（转换为base64）
        print(f"🖼️  请求 {request_id}: 正在加载图片...")
        image_data = self.encode_image_to_base64(image_url)

        if not image_data:
            return {
                'request_id': request_id,
                'success': False,
                'error': 'Failed to load image',
                'total_time': 0,
                'tokens_per_second': 0,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }

        # 构建请求消息（支持多种格式）
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data  # 使用base64编码的图片
                        }
                    },
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        ]

        payload = {
            # "model": "Qwen/Qwen3-VL-4B-Instruct",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }

        start_time = time.time()
        print(f"🚀 请求 {request_id}: 开始处理...")

        try:
            async with session.post(f"{self.base_url}/v1/chat/completions", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    end_time = time.time()

                    # 计算性能指标
                    total_time = end_time - start_time

                    # 获取生成的文本和token数量
                    generated_text = result['choices'][0]['message']['content']

                    # 获取token使用情况
                    usage = result.get('usage', {})
                    completion_tokens = usage.get('completion_tokens', len(generated_text.split()))
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    total_tokens = usage.get('total_tokens', completion_tokens + prompt_tokens)

                    # 计算生成速度
                    tokens_per_second = completion_tokens / total_time if total_time > 0 else 0

                    return {
                        'request_id': request_id,
                        'success': True,
                        'image_url': image_url,
                        'text_prompt': text_prompt,
                        'response': generated_text,
                        'total_time': total_time,
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': total_tokens,
                        'tokens_per_second': tokens_per_second,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    }
                else:
                    error_text = await response.text()
                    return {
                        'request_id': request_id,
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}",
                        'total_time': time.time() - start_time,
                        'tokens_per_second': 0,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    }

        except Exception as e:
            return {
                'request_id': request_id,
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time,
                'tokens_per_second': 0,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }

    async def run_vision_concurrent_requests(self, test_cases: List[Dict], max_tokens: int = 512):
        """并发运行多个视觉理解请求"""

        print(f"🚀 开始视觉模型并发测试...")
        print(f"📊 请求数量: {len(test_cases)}")
        print(f"🎯 最大token数: {max_tokens}")
        print(f"🌐 服务器地址: {self.base_url}")
        print("-" * 80)

        connector = aiohttp.TCPConnector(limit=10)
        timeout = aiohttp.ClientTimeout(total=600)  # 10分钟超时（图片处理需要更长时间）

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # 创建并发任务
            tasks = [
                self.send_vision_request(
                    session,
                    i,
                    case['image_url'],
                    case['text_prompt'],
                    max_tokens
                )
                for i, case in enumerate(test_cases, 1)
            ]

            # 等待所有任务完成
            overall_start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            overall_end = time.time()

            # 处理结果
            self.results = []
            for result in results:
                if isinstance(result, Exception):
                    self.results.append({
                        'success': False,
                        'error': str(result),
                        'tokens_per_second': 0
                    })
                else:
                    self.results.append(result)

            # 显示结果
            self.display_vision_results(overall_end - overall_start)

    def display_vision_results(self, total_time: float):
        """显示视觉测试结果"""

        print("\n" + "=" * 100)
        print("📋 视觉理解测试详细结果:")
        print("=" * 100)

        successful_requests = []
        failed_requests = []

        for result in self.results:
            if result['success']:
                successful_requests.append(result)
                print(f"✅ 请求 {result['request_id']} [{result['timestamp']}]")
                print(f"   🖼️  图片: {result['image_url']}")
                print(f"   📝 提示: {result['text_prompt']}")
                print(f"   ⏱️  耗时: {result['total_time']:.2f}s")
                print(f"   🎯 生成Token: {result['completion_tokens']} 个")
                print(f"   ⚡ 速度: {result['tokens_per_second']:.2f} tokens/s")
                print(f"   💬 回复: {result['response'][:800]}{'...' if len(result['response']) > 800 else ''}")
                print("-" * 80)
            else:
                failed_requests.append(result)
                print(f"❌ 请求 {result.get('request_id', '?')} 失败: {result.get('error', 'Unknown error')}")
                print()

        # 统计信息
        print("=" * 100)
        print("📊 视觉模型性能统计:")
        print("=" * 100)

        if successful_requests:
            # 计算各种统计指标
            response_times = [r['total_time'] for r in successful_requests]
            tokens_per_second = [r['tokens_per_second'] for r in successful_requests if r['tokens_per_second'] > 0]
            total_tokens = sum(r['completion_tokens'] for r in successful_requests)
            total_prompt_tokens = sum(r['prompt_tokens'] for r in successful_requests)

            print(f"✅ 成功请求: {len(successful_requests)}/{len(self.results)}")
            print(f"❌ 失败请求: {len(failed_requests)}")
            print(f"⏱️  总耗时: {total_time:.2f}s")
            print(f"🎯 总生成Token: {total_tokens}")
            print(f"📝 总输入Token: {total_prompt_tokens}")
            print()

            if response_times:
                print("⏱️  响应时间统计:")
                print(f"   平均: {statistics.mean(response_times):.2f}s")
                print(f"   最快: {min(response_times):.2f}s")
                print(f"   最慢: {max(response_times):.2f}s")
                print(f"   中位数: {statistics.median(response_times):.2f}s")
                print()

            if tokens_per_second:
                print("⚡ Token生成速度统计:")
                print(f"   平均: {statistics.mean(tokens_per_second):.2f} tokens/s")
                print(f"   最快: {max(tokens_per_second):.2f} tokens/s")
                print(f"   最慢: {min(tokens_per_second):.2f} tokens/s")
                print(f"   中位数: {statistics.median(tokens_per_second):.2f} tokens/s")
                print()

            # 整体吞吐量
            overall_throughput = total_tokens / total_time if total_time > 0 else 0
            print(f"🚀 整体吞吐量: {overall_throughput:.2f} tokens/s")
            print(f"📈 并发效率: {len(successful_requests) / total_time:.2f} requests/s")
            print(f"🖼️  图片处理效率: {len(successful_requests) / total_time:.2f} images/s")

        else:
            print("❌ 没有成功的请求")

        print("=" * 100)


def get_vision_test_cases() -> List[Dict]:
    """获取视觉测试用例"""
    return [
        {
            "image_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            "text_prompt": "Describe this image in detail."
        },
        {
            "image_url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png",
            "text_prompt": "Read all the text in the image."
        },
        {
            "image_url": "https://language.chinadaily.com.cn/images/attachement/jpg/site1/20160510/00221910993f189bf0bc52.jpg",
            "text_prompt": "Extract the full text from the newspaper exactly as it is, including every word."
        },
        {
            "image_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            "text_prompt": "What objects can you see in this image? List them."
        },
        {
            "image_url": "https://img2.baidu.com/it/u=3143282702,2145280608&fm=253&app=138&f=JPEG?w=800&h=1176",
            "text_prompt": "What characters do you see in this image?"
        },
        {
            "image_url": "https://img1.baidu.com/it/u=3809733790,1132788165&fm=253&app=138&f=JPEG?w=1422&h=800",
            "text_prompt": "What is the weather like in this image?"
        },
        {
            "image_url": "https://img1.baidu.com/it/u=4225257766,2460304301&fm=253&fmt=auto&app=138&f=JPEG?w=231&h=500",
            "text_prompt": "What character do you see in this image, and what mood does the character feel?"
        },
        {
            "image_url": "https://img2.baidu.com/it/u=500643173,2012953556&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=1407",
            "text_prompt": "What characters do you see in this image and what might have happened between them?"
        },
        {
            "image_url": "https://h.cdn.zhuolaoshi.cn/user/site26644/image/20180927/20180927072723952395.jpg",
            "text_prompt": "What information can you extract from this image? "
                           "Please organize the extracted information into a structured format and present it."
        },
        {
            "image_url": "https://cje.ustb.edu.cn/fileGCKXXB/journal/article/gckxxb/2023/10/230315-0003-3.jpg",
            "text_prompt": "Analyze what you see in the picture."
        }
    ]


async def main():
    parser = argparse.ArgumentParser(description="vLLM 视觉模型并发性能测试工具")
    parser.add_argument("--url", default="http://localhost:8001", help="vLLM 服务器地址")
    parser.add_argument("--requests", "-r", type=int, default=3, help="并发请求数量")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成token数")
    parser.add_argument("--image-url", help="自定义图片URL")
    parser.add_argument("--text-prompt", help="自定义文本提示")

    args = parser.parse_args()

    # 准备测试用例
    if args.image_url and args.text_prompt:
        # 使用自定义的图片和提示
        test_cases = [
                         {
                             "image_url": args.image_url,
                             "text_prompt": args.text_prompt
                         }
                     ] * args.requests
    else:
        # 使用预定义的测试用例
        base_cases = get_vision_test_cases()
        test_cases = (base_cases * ((args.requests // len(base_cases)) + 1))[:args.requests]

    # 创建客户端并运行测试
    client = VisionConcurrentClient(args.url)
    await client.run_vision_concurrent_requests(test_cases, args.max_tokens)


if __name__ == "__main__":
    asyncio.run(main())
