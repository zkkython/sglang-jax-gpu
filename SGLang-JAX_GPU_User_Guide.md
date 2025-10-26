# SGLang-JAX GPU User Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Server Setup](#server-setup)
5. [Running the Server](#running-the-server)
6. [Benchmarking](#benchmarking)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

## Overview

SGLang-JAX is a high-performance inference server for large language models using JAX/XLA backend. This guide covers how to run SGLang-JAX on GPU systems with CUDA support.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX 3080/4080, A100, H100, etc.)
- **Memory**: Minimum 16GB GPU memory (recommended 24GB+ for larger models)
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 32GB+ system memory

### Software Requirements
- **CUDA**: Version 11.8 or higher
- **Python**: 3.8-3.12
- **JAX**: Latest version with CUDA support
- **CUDA Toolkit**: Compatible with your GPU

## Installation

### 1. Create Conda Environment

```bash
# Create a new conda environment
conda create -n jax_gpu python=3.12
conda activate jax_gpu

# Install CUDA toolkit (if not already installed)
conda install cuda-toolkit=11.8
```

### 2. Install JAX with CUDA Support

```bash
# Install JAX with CUDA support
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify JAX installation
python -c "import jax; print(jax.devices())"
```

### 3. Install SGLang-JAX

```bash
# Clone the repository
git clone https://github.com/sgl-project/sglang-jax.git
cd sglang-jax

# Install in development mode
pip install -e .
```

### 4. Install Additional Dependencies

```bash
# Install required packages
pip install transformers torch numpy scipy
pip install fastapi uvicorn zmq
pip install psutil setproctitle
```

## Server Setup

### 1. Model Preparation

Download your model to a local directory:

```bash
# Example: Download Qwen model
mkdir -p /home/user/models/qwen06b
# Download model files to this directory
```

### 2. Configuration Files

Create a server configuration file `server_config.yaml`:

```yaml
model_path: "/home/user/models/qwen06b"
tokenizer_path: "/home/user/models/qwen06b"
host: "0.0.0.0"
port: 30000
dtype: "bfloat16"
device: "cuda"
attention_backend: "native"
xla_backend: "native"
mem_fraction_static: 0.8
max_prefill_tokens: 8192
max_running_requests: 100
trust_remote_code: true
skip_server_warmup: true
```

## Running the Server

### 1. Basic Server Startup

```bash
# Activate environment
conda activate jax_gpu

# Set JAX compilation cache directory
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache

# Start the server
python -u -m sgl_jax.launch_server \
    --model-path /home/user/models/qwen06b \
    --trust-remote-code \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1 \
    --tp-size=1 \
    --xla-backend=native \
    --attention-backend=native \
    --device=cuda \
    --random-seed=3 \
    --node-rank=0 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=8192 \
    --download-dir=/tmp \
    --dtype=bfloat16 \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port 30000
```

### 2. Memory-Optimized Configuration

For GPUs with limited memory:

```bash
python -u -m sgl_jax.launch_server \
    --model-path /home/user/models/qwen06b \
    --trust-remote-code \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1 \
    --tp-size=1 \
    --xla-backend=native \
    --attention-backend=native \
    --device=cuda \
    --random-seed=3 \
    --node-rank=0 \
    --mem-fraction-static=0.6 \
    --max-prefill-tokens=4096 \
    --max-running-requests=50 \
    --download-dir=/tmp \
    --dtype=bfloat16 \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port 30000
```

### 3. High-Performance Configuration

For high-end GPUs:

```bash
python -u -m sgl_jax.launch_server \
    --model-path /home/user/models/qwen06b \
    --trust-remote-code \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1 \
    --tp-size=1 \
    --xla-backend=native \
    --attention-backend=native \
    --device=cuda \
    --random-seed=3 \
    --node-rank=0 \
    --mem-fraction-static=0.9 \
    --max-prefill-tokens=16384 \
    --max-running-requests=200 \
    --download-dir=/tmp \
    --dtype=bfloat16 \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port 30000
```

## Benchmarking

### 1. Basic Benchmark Script

Create `benchmark.py`:

```python
import asyncio
import aiohttp
import time
import json
import argparse
from typing import List, Dict, Any

class SGLangBenchmark:
    def __init__(self, base_url: str = "http://localhost:30000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_request(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Send a single generation request"""
        payload = {
            "model": "qwen06b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }
        
        start_time = time.time()
        async with self.session.post(f"{self.base_url}/v1/chat/completions", 
                                   json=payload) as response:
            result = await response.json()
            end_time = time.time()
            
            return {
                "response_time": end_time - start_time,
                "tokens_generated": len(result.get("choices", [{}])[0].get("message", {}).get("content", "").split()),
                "success": response.status == 200,
                "status_code": response.status
            }
    
    async def run_benchmark(self, prompts: List[str], concurrent_requests: int = 10, 
                          max_tokens: int = 100) -> Dict[str, Any]:
        """Run benchmark with concurrent requests"""
        print(f"Running benchmark with {concurrent_requests} concurrent requests...")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def limited_request(prompt):
            async with semaphore:
                return await self.generate_request(prompt, max_tokens)
        
        # Run all requests
        start_time = time.time()
        tasks = [limited_request(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Process results
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_results = [r for r in results if not isinstance(r, dict) or not r.get("success")]
        
        total_time = end_time - start_time
        total_requests = len(prompts)
        successful_requests = len(successful_results)
        
        if successful_results:
            avg_response_time = sum(r["response_time"] for r in successful_results) / len(successful_results)
            total_tokens = sum(r["tokens_generated"] for r in successful_results)
            tokens_per_second = total_tokens / total_time
            requests_per_second = successful_requests / total_time
        else:
            avg_response_time = 0
            total_tokens = 0
            tokens_per_second = 0
            requests_per_second = 0
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": len(failed_results),
            "success_rate": successful_requests / total_requests * 100,
            "total_time": total_time,
            "avg_response_time": avg_response_time,
            "requests_per_second": requests_per_second,
            "tokens_per_second": tokens_per_second,
            "total_tokens_generated": total_tokens
        }

async def main():
    parser = argparse.ArgumentParser(description="SGLang-JAX Benchmark Tool")
    parser.add_argument("--url", default="http://localhost:30000", help="Server URL")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--requests", type=int, default=100, help="Total requests")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per request")
    parser.add_argument("--prompt-file", help="File containing prompts (one per line)")
    
    args = parser.parse_args()
    
    # Generate or load prompts
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default prompts
        prompts = [
            "Explain the concept of machine learning in simple terms.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does blockchain technology work?",
            "What is the importance of data privacy?"
        ] * (args.requests // 5 + 1)
    
    prompts = prompts[:args.requests]
    
    async with SGLangBenchmark(args.url) as benchmark:
        print(f"Starting benchmark with {len(prompts)} requests...")
        results = await benchmark.run_benchmark(prompts, args.concurrent, args.max_tokens)
        
        print("\n=== Benchmark Results ===")
        print(f"Total Requests: {results['total_requests']}")
        print(f"Successful Requests: {results['successful_requests']}")
        print(f"Failed Requests: {results['failed_requests']}")
        print(f"Success Rate: {results['success_rate']:.2f}%")
        print(f"Total Time: {results['total_time']:.2f}s")
        print(f"Average Response Time: {results['avg_response_time']:.3f}s")
        print(f"Requests per Second: {results['requests_per_second']:.2f}")
        print(f"Tokens per Second: {results['tokens_per_second']:.2f}")
        print(f"Total Tokens Generated: {results['total_tokens_generated']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Load Testing Script

Create `load_test.py`:

```python
import asyncio
import aiohttp
import time
import json
import argparse
import statistics
from typing import List, Dict, Any
import matplotlib.pyplot as plt

class LoadTester:
    def __init__(self, base_url: str = "http://localhost:30000"):
        self.base_url = base_url
        self.session = None
        self.results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def single_request(self, prompt: str, request_id: int) -> Dict[str, Any]:
        """Send a single request and measure performance"""
        payload = {
            "model": "qwen06b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.7,
            "stream": False
        }
        
        start_time = time.time()
        try:
            async with self.session.post(f"{self.base_url}/v1/chat/completions", 
                                       json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                result = await response.json()
                end_time = time.time()
                
                return {
                    "request_id": request_id,
                    "response_time": end_time - start_time,
                    "status_code": response.status,
                    "success": response.status == 200,
                    "timestamp": start_time,
                    "tokens": len(result.get("choices", [{}])[0].get("message", {}).get("content", "").split())
                }
        except Exception as e:
            end_time = time.time()
            return {
                "request_id": request_id,
                "response_time": end_time - start_time,
                "status_code": 0,
                "success": False,
                "timestamp": start_time,
                "error": str(e),
                "tokens": 0
            }
    
    async def run_load_test(self, duration: int = 60, rps: int = 10) -> Dict[str, Any]:
        """Run load test for specified duration"""
        print(f"Starting load test: {duration}s duration, {rps} requests/second")
        
        start_time = time.time()
        request_id = 0
        tasks = []
        
        # Generate requests at specified rate
        while time.time() - start_time < duration:
            batch_start = time.time()
            
            # Create batch of requests
            batch_tasks = []
            for _ in range(rps):
                prompt = f"Request {request_id}: Explain quantum computing briefly."
                batch_tasks.append(self.single_request(prompt, request_id))
                request_id += 1
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            self.results.extend([r for r in batch_results if isinstance(r, dict)])
            
            # Wait for next second
            batch_time = time.time() - batch_start
            if batch_time < 1.0:
                await asyncio.sleep(1.0 - batch_time)
        
        # Wait for remaining requests to complete
        await asyncio.sleep(5)
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        successful_results = [r for r in self.results if r.get("success")]
        failed_results = [r for r in self.results if not r.get("success")]
        
        if successful_results:
            response_times = [r["response_time"] for r in successful_results]
            tokens = [r["tokens"] for r in successful_results]
            
            analysis = {
                "total_requests": len(self.results),
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": len(successful_results) / len(self.results) * 100,
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)],
                "p99_response_time": sorted(response_times)[int(len(response_times) * 0.99)],
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "total_tokens": sum(tokens),
                "avg_tokens_per_request": statistics.mean(tokens),
                "tokens_per_second": sum(tokens) / (self.results[-1]["timestamp"] - self.results[0]["timestamp"]) if len(self.results) > 1 else 0
            }
        else:
            analysis = {
                "total_requests": len(self.results),
                "successful_requests": 0,
                "failed_requests": len(failed_results),
                "success_rate": 0,
                "error": "All requests failed"
            }
        
        return analysis
    
    def plot_results(self, output_file: str = "load_test_results.png"):
        """Plot response time distribution"""
        if not self.results:
            return
        
        successful_results = [r for r in self.results if r.get("success")]
        if not successful_results:
            return
        
        response_times = [r["response_time"] for r in successful_results]
        timestamps = [r["timestamp"] for r in successful_results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Response time over time
        ax1.plot(timestamps, response_times, 'b-', alpha=0.7)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Response Time (s)')
        ax1.set_title('Response Time Over Time')
        ax1.grid(True)
        
        # Response time histogram
        ax2.hist(response_times, bins=50, alpha=0.7, color='green')
        ax2.set_xlabel('Response Time (s)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Response Time Distribution')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")

async def main():
    parser = argparse.ArgumentParser(description="SGLang-JAX Load Tester")
    parser.add_argument("--url", default="http://localhost:30000", help="Server URL")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--rps", type=int, default=10, help="Requests per second")
    parser.add_argument("--plot", action="store_true", help="Generate performance plots")
    
    args = parser.parse_args()
    
    async with LoadTester(args.url) as tester:
        results = await tester.run_load_test(args.duration, args.rps)
        
        print("\n=== Load Test Results ===")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        
        if args.plot:
            tester.plot_results()

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Running Benchmarks

```bash
# Basic benchmark
python benchmark.py --concurrent 10 --requests 100

# Load test
python load_test.py --duration 120 --rps 20 --plot

# Custom prompts
echo "What is artificial intelligence?" > prompts.txt
echo "Explain machine learning" >> prompts.txt
python benchmark.py --prompt-file prompts.txt --concurrent 5 --requests 50
```

## Troubleshooting

### Common Issues and Solutions

#### 1. PyTreeDef Duplicate Registration Error

**Error**: `ValueError: Duplicate custom PyTreeDef type registration`

**Solution**: This is already fixed in the current codebase. If you encounter this error, ensure you're using the latest version.

#### 2. GPU Memory Issues

**Error**: `RESOURCE_EXHAUSTED: Out of memory`

**Solutions**:
```bash
# Reduce memory usage
--mem-fraction-static=0.4
--max-prefill-tokens=2048
--max-running-requests=10

# Check GPU memory
nvidia-smi
```

#### 3. CUDA Compatibility Issues

**Error**: CUDA version mismatch

**Solutions**:
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Reinstall JAX with correct CUDA version
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### 4. Port Already in Use

**Error**: `Address already in use`

**Solutions**:
```bash
# Find and kill process using port 30000
lsof -ti:30000 | xargs kill -9

# Use different port
--port 30001
```

#### 5. Model Loading Issues

**Error**: Model not found or loading failed

**Solutions**:
```bash
# Check model path
ls -la /home/user/models/qwen06b/

# Verify model files
--trust-remote-code
--load-format=auto
```

### Debugging Commands

```bash
# Enable debug logging
export JAX_TRACEBACK_FILTERING=off
export CUDA_VISIBLE_DEVICES=0

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check server logs
tail -f server.log

# Test server health
curl http://localhost:30000/health
```

## Performance Optimization

### 1. JAX Compilation Cache

```bash
# Set persistent cache directory
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
mkdir -p /tmp/jit_cache

# Enable all caches
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all
```

### 2. Memory Optimization

```bash
# Optimize memory allocation
--mem-fraction-static=0.8  # Use 80% of GPU memory
--max-prefill-tokens=8192  # Limit prefill tokens
--max-running-requests=100  # Limit concurrent requests
```

### 3. Batch Size Tuning

```bash
# For high throughput
--max-running-requests=200
--max-prefill-tokens=4096

# For low latency
--max-running-requests=50
--max-prefill-tokens=2048
```

### 4. Hardware-Specific Optimizations

#### RTX 4090 (24GB)
```bash
--mem-fraction-static=0.9
--max-prefill-tokens=16384
--max-running-requests=150
```

#### A100 (40GB)
```bash
--mem-fraction-static=0.95
--max-prefill-tokens=32768
--max-running-requests=300
```

#### RTX 3080 (10GB)
```bash
--mem-fraction-static=0.7
--max-prefill-tokens=4096
--max-running-requests=50
```

### 5. Monitoring Performance

```bash
# Real-time monitoring script
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
    echo "Server status:"
    curl -s http://localhost:30000/health || echo "Server not responding"
    echo ""
    sleep 5
done
```

## Advanced Configuration

### 1. Multi-GPU Setup

```bash
# For multi-GPU systems
--tp-size=2  # Tensor parallelism across 2 GPUs
--device=cuda:0,1  # Use specific GPUs
```

### 2. Custom Model Configuration

```python
# Custom model configuration
model_config = {
    "model_path": "/path/to/model",
    "dtype": "bfloat16",
    "max_seq_len": 4096,
    "attention_backend": "native",
    "xla_backend": "native"
}
```

### 3. Production Deployment

```bash
# Production startup script
#!/bin/bash
export JAX_COMPILATION_CACHE_DIR=/opt/sglang/cache
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

python -u -m sgl_jax.launch_server \
    --model-path /opt/models/qwen06b \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=8192 \
    --max-running-requests=100 \
    --dtype=bfloat16 \
    --trust-remote-code \
    --skip-server-warmup \
    --log-level=info
```

## Conclusion

This guide provides comprehensive instructions for running SGLang-JAX on GPU systems. The key to successful deployment is proper configuration based on your hardware specifications and workload requirements. Always monitor performance metrics and adjust parameters accordingly for optimal results.

For additional support and updates, refer to the official SGLang-JAX documentation and GitHub repository.
