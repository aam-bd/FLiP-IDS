#!/bin/bash

# Script to run the IoT Security Framework API server
# 
# Provides options for development and production deployment

set -e  # Exit on any error

echo "🛡️  Starting IoT Security Framework API Server"
echo "=============================================="
echo ""

# Default configuration
HOST="0.0.0.0"
PORT="8000"
WORKERS=1
LOG_LEVEL="info"
RELOAD=false
ENV="development"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --reload)
            RELOAD=true
            shift
            ;;
        --production)
            ENV="production"
            WORKERS=4
            LOG_LEVEL="warning"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST         Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT         Port to bind to (default: 8000)"
            echo "  --workers N         Number of worker processes (default: 1)"
            echo "  --log-level LEVEL   Log level (default: info)"
            echo "  --reload            Enable auto-reload for development"
            echo "  --production        Production mode (4 workers, warning logs)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Development server"
            echo "  $0 --reload                     # Development with auto-reload"
            echo "  $0 --production                 # Production server"
            echo "  $0 --host 127.0.0.1 --port 9000  # Custom host/port"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]] && [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider activating a virtual environment first:"
    echo "   python -m venv venv && source venv/bin/activate"
    echo ""
fi

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
if ! python -c "import fastapi, torch, sklearn, pandas" 2>/dev/null; then
    echo "❌ Missing dependencies. Please install requirements:"
    echo "   pip install -r requirements.txt"
    exit 1
fi
echo "✅ Dependencies check passed"
echo ""

# Create necessary directories
echo "📁 Setting up directories..."
mkdir -p data/{raw,interim,processed,models}
mkdir -p logs
mkdir -p checkpoints
echo "✅ Directories created"
echo ""

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export IOT_FRAMEWORK_ENV="$ENV"

# Display configuration
echo "⚙️  Server Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Workers: $WORKERS"
echo "   Log Level: $LOG_LEVEL"
echo "   Reload: $RELOAD"
echo "   Environment: $ENV"
echo ""

# Check if port is available
if command -v lsof >/dev/null 2>&1; then
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null; then
        echo "⚠️  Port $PORT is already in use!"
        echo "   Use --port to specify a different port"
        exit 1
    fi
fi

# Start server
echo "🚀 Starting server..."
echo "   API Documentation: http://$HOST:$PORT/docs"
echo "   Alternative Docs: http://$HOST:$PORT/redoc"
echo "   Health Check: http://$HOST:$PORT/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Build uvicorn command
UVICORN_CMD="uvicorn apps.service:app"
UVICORN_CMD="$UVICORN_CMD --host $HOST"
UVICORN_CMD="$UVICORN_CMD --port $PORT"
UVICORN_CMD="$UVICORN_CMD --log-level $LOG_LEVEL"

if [[ "$RELOAD" == "true" ]]; then
    UVICORN_CMD="$UVICORN_CMD --reload"
else
    UVICORN_CMD="$UVICORN_CMD --workers $WORKERS"
fi

# Add access logging for production
if [[ "$ENV" == "production" ]]; then
    UVICORN_CMD="$UVICORN_CMD --access-log"
fi

# Execute the command
exec $UVICORN_CMD
