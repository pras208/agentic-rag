#!/bin/bash

# Quick start script for Agentic RAG POC

set -e

echo "🚀 Agentic RAG POC - Quick Start"
echo "=================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -q -r requirements.txt

# Check AWS credentials
echo "🔐 Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "⚠️  AWS credentials not configured"
    echo "   Run: aws configure"
    echo "   Or set AWS_PROFILE environment variable"
fi

# Create directories
mkdir -p uploads indexes

# Setup .env if not exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "   Edit .env to configure AWS region if needed"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 To start the server:"
echo "   python app.py"
echo ""
echo "🌐 Then open:"
echo "   http://localhost:5000"
echo ""
echo "📄 Try uploading sample.md as a test document"
