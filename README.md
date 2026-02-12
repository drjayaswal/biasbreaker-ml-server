---
title: Alluvium ML Server
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

# Alluvium ML Server

A machine learning server for the Alluvium project that detects and mitigates bias in AI systems.

## Prerequisites

- Python (v3.8 or higher)
- pip or conda
- Git

## Installation

```bash
# Clone the repository
git clone https://github.com/drjayaswal/alluvium-ml-server.git

# Navigate to the project directory
cd alluvium-ml-server

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup

```bash
# Create a .env file
cp .env.example .env

# Update .env with your configuration
```

## Running the ML Server

```bash
# Development mode with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

## Git Workflow

To stage all changes for a commit:
```bash
git add .
```

To commit changes with a message:
```bash
git commit -m "describe your changes"
```

## Docker Deployment

To build the ML server Docker image:
```bash
docker build -t dhruv2k3/alluvium-ml-server:latest .
```

To test locally:
```bash
docker run -p 8001:8001 dhruv2k3/alluvium-ml-server
```

To push to Docker Hub:
```bash
docker push dhruv2k3/alluvium-ml-server:latest
```

To run via docker-compose:
```bash
docker compose pull
docker compose up -d
```

## API Documentation

Visit `http://localhost:8001/docs` for API endpoints documentation.

## Contributing

Contributions are welcome. Please follow the existing code style and submit pull requests to the main branch.

