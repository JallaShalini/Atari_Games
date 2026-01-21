# Testing Guide for Atari DQN Agent

This document provides comprehensive instructions for testing all commands in the submission.yml file.

## Prerequisites

1. Docker and Docker Compose installed
2. Python 3.9+
3. At least 4GB of RAM for Docker operations
4. CUDA/GPU (optional but recommended for faster training)
5. Linux/Mac/Windows with WSL2 for Docker

## Quick Test Checklist

- [ ] Test 1: Docker Build Commands
- [ ] Test 2: Environment Validation  
- [ ] Test 3: Training Command
- [ ] Test 4: Evaluation Command
- [ ] Test 5: Play/Inference Command
- [ ] Test 6: API Server
- [ ] Test 7: Unit Tests

## Detailed Testing Instructions

### Test 1: Docker Build Commands

```bash
# Build training Docker image
docker build -t rl-agent-train -f docker/Dockerfile.train .

# Verify build succeeded
docker images | grep rl-agent-train

# Build inference Docker image  
docker build -t rl-agent-inference -f docker/Dockerfile.inference .

# Verify build succeeded
docker images | grep rl-agent-inference
```

**Expected:** Both images build successfully without errors

### Test 2: Environment Validation

```bash
# Run sanity check
python scripts/sanity_check_env.py

# Check core dependencies
python -c "import torch; import gym; import numpy; print('All dependencies OK')"
```

**Expected:** All environment checks pass

### Test 3: Training Command

```bash
# Create models directory
mkdir -p models

# Run training inside Docker
docker run --rm -v $(pwd)/models:/app/models rl-agent-train python scripts/train.py --game Pong-v5 --episodes 10

# For local testing (without Docker):
python scripts/train.py --game Pong-v5 --episodes 5
```

**Expected:** 
- Training starts and processes frames
- Model checkpoints saved to models/ directory  
- No errors in execution
- Training completes in reasonable time

### Test 4: Evaluation Command

```bash
# Run evaluation inside Docker
docker run --rm -v $(pwd)/models:/app/models rl-agent-train python scripts/evaluate.py --game Pong-v5 --episodes 10

# For local testing:
python scripts/evaluate.py --game Pong-v5 --episodes 5
```

**Expected:**
- Evaluation runs against trained model
- Average reward metric computed
- Output contains "AVERAGE_REWARD" line
- Example output: "AVERAGE_REWARD: 12.5"

### Test 5: Play Command (Video Generation)

```bash
# Create videos directory
mkdir -p videos

# Run play command inside Docker
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/videos:/app/videos rl-agent-train python scripts/play.py --game Pong-v5 --episodes 2

# For local testing:
python scripts/play.py --game Pong-v5 --episodes 1
```

**Expected:**
- Agent plays game
- Video file generated in videos/ directory
- Video shows agent playing Pong
- No errors during playback

### Test 6: Full Submission.yml Command Sequence

```bash
# Test build
docker build -t rl-agent-train -f docker/Dockerfile.train .
docker build -t rl-agent-inference -f docker/Dockerfile.inference .

# Test train
mkdir -p models
docker run --rm -v $(pwd)/models:/app/models rl-agent-train python scripts/train.py --game Pong-v5 --episodes 100

# Test evaluate
docker run --rm -v $(pwd)/models:/app/models rl-agent-train python scripts/evaluate.py --game Pong-v5 --episodes 100

# Test play
mkdir -p videos
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/videos:/app/videos rl-agent-train python scripts/play.py --game Pong-v5 --episodes 5
```

### Test 7: API Server Testing

```bash
# Run API server in Docker
docker run --rm -p 8000:8000 rl-agent-inference

# In another terminal, test endpoints:

# Health check
curl http://localhost:8000/health

# Prediction request (example with dummy state)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"state": [[[[0.0]]]*84]*4}'
```

**Expected:**
- API server starts and listens on port 8000
- Health endpoint returns {"status": "healthy"}
- Predict endpoint accepts state input and returns action + Q-values

### Test 8: Unit Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_main.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**Expected:**
- All tests pass
- Code coverage reasonable (>80%)
- No import or runtime errors

## Troubleshooting

### Issue: Docker build fails
**Solution:** Ensure all required files exist (Dockerfile.train, Dockerfile.inference, requirements.txt, src/, configs/)

### Issue: Out of memory during training
**Solution:** Reduce --episodes or add GPU support with `docker run --gpus all`

### Issue: Module not found errors
**Solution:** Ensure all Python packages in requirements.txt are installed

### Issue: Video file not generated
**Solution:** Check videos/ directory permissions and ensure pygame/ffmpeg is installed

## Performance Expectations

- Build time: 2-5 minutes per image
- Training (100 episodes): 15-30 minutes (CPU) or 5-10 minutes (GPU)
- Evaluation (100 episodes): 10-20 minutes  
- Play (5 episodes): 2-5 minutes
- Average Reward Target: >= 10 on Pong-v5

## Success Criteria

- All Docker images build without errors
- Training command completes successfully
- Evaluation produces AVERAGE_REWARD metric >= 10
- Video file generated successfully
- API server runs and responds to health checks
- All unit tests pass

## Command Reference

From submission.yml:

```yaml
build:
  command: |
    docker build -t rl-agent-train -f docker/Dockerfile.train .
    docker build -t rl-agent-inference -f docker/Dockerfile.inference .

train:
  command: |
    docker run --rm -v $(pwd)/models:/app/models rl-agent-train python scripts/train.py --game Pong-v5 --episodes 100

evaluate:
  command: |
    docker run --rm -v $(pwd)/models:/app/models rl-agent-train python scripts/evaluate.py --game Pong-v5 --episodes 100
  output_parser: "AVERAGE_REWARD"

play:
  command: |
    docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/videos:/app/videos rl-agent-train python scripts/play.py --game Pong-v5 --episodes 5
```

## Next Steps

1. Clone the repository
2. Follow testing instructions sequentially
3. Record results for each test
4. Submit project once all tests pass
