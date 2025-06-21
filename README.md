# HFT-RAG System

High-Frequency Trading RAG System for market data analysis and strategy augmentation.

## Ubuntu Deployment Guide

### System Requirements
- **Hardware**: i7 12700H + RTX 3050 Ti (16GB RAM minimum)
- **OS**: Ubuntu 22.04 LTS or later
- **GPU**: NVIDIA GPU with CUDA support

### Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget git build-essential pkg-config libssl-dev ca-certificates

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Docker & Docker Compose
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER
sudo systemctl enable docker
sudo systemctl start docker

# Install NVIDIA CUDA Toolkit (for GPU acceleration)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-3
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Reboot system to activate CUDA
sudo reboot
```

### Quick Start (Docker - Recommended)

```bash
# Clone repository
git clone <repository-url>
cd hft-rag

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f hft-rag

# Health check
curl http://localhost:8080/health
```

### Native Installation

```bash
# Build application
cargo build --release

# Set environment variables
export RUST_LOG=info
export HFT_RAG_SERVER_HOST=0.0.0.0
export HFT_RAG_SERVER_PORT=8080
export HFT_RAG_STORAGE_QDRANT_URL=http://localhost:6333

# Start Qdrant (in separate terminal)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Run application
./target/release/hft-rag
```

### Performance Optimization

```bash
# Enable GPU memory optimization
echo 'CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
echo 'CUDA_CACHE_MAXSIZE=2147483648' >> ~/.bashrc

# Set CPU governor for performance
sudo cpupower frequency-set -g performance

# Optimize network settings for HFT
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Verification

```bash
# Test GPU acceleration
nvidia-smi

# Test embedding service
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "AAPL price movement analysis"}'

# Test query processing
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest AAPL trading patterns?"}'
```

### Production Deployment

```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d

# Enable automatic startup
sudo systemctl enable docker
```

### Troubleshooting

**GPU not detected:**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Restart Docker with NVIDIA runtime
sudo systemctl restart docker
```

**Port conflicts:**
```bash
# Check port usage
sudo netstat -tulpn | grep :8080
sudo netstat -tulpn | grep :6333
```

**Memory issues:**
```bash
# Monitor memory usage
htop
docker stats
```

## API Endpoints

- `GET /health` - Health check
- `POST /query` - Query processing
- `POST /embed` - Text embedding
- `POST /ingest` - Document ingestion

## Configuration

Environment variables:
- `HFT_RAG_SERVER_HOST` - Server host (default: 0.0.0.0)
- `HFT_RAG_SERVER_PORT` - Server port (default: 8080)
- `HFT_RAG_STORAGE_QDRANT_URL` - Qdrant URL
- `HFT_RAG_EMBEDDING_PROVIDER` - Embedding provider (candle/openai)
- `RUST_LOG` - Log level (info/debug/error)