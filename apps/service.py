"""
Main FastAPI service for the IoT Security Framework.

Provides a unified API service that mounts both Phase 1 (Device Profiling)
and Phase 2 (Federated Learning IDS) endpoints, along with system monitoring,
health checks, and documentation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.phase1_profiling.api import router as phase1_router
from src.phase2_ids.api import router as phase2_router
from src.common.logging import setup_logging, get_logger
from src.common.schemas import APIResponse
from src.common.utils import get_device, memory_usage, disk_usage

# Setup logging
setup_logging(level="INFO", include_timestamp=True)
logger = get_logger(__name__)

# Application metadata
APP_NAME = "IoT Security Framework"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = """
# IoT Security Framework API

A comprehensive two-phase framework for securing heterogeneous IoT networks:

## Phase 1: Network Discovery and Device Profiling
- **PCAP Processing**: Extract features from network traffic
- **Device Classification**: IoT vs Non-IoT identification
- **Device Profiling**: Specific device type classification
- **Feature Selection**: Hybrid feature extraction and selection

## Phase 2: Self-Labeled Federated Learning IDS  
- **Federated Learning**: Collaborative intrusion detection
- **Meta-Learning**: MAML-based personalization
- **Privacy-Preserving**: CT-AE encoding for similarity computation
- **Self-Labeling**: Automatic attack annotation via BS-Agg

## Key Features
- 🔒 **Privacy-First**: Raw data stays local, only encoded features shared
- 🤖 **Self-Learning**: Zero-day attack detection without manual labeling  
- 🌐 **Federated**: Collaborative learning across IoT gateways
- 📊 **Comprehensive**: End-to-end pipeline from PCAP to detection
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    logger.info(f"Device: {get_device()}")
    
    # Log system info
    mem_info = memory_usage()
    if mem_info:
        logger.info(f"Memory usage: {mem_info['rss_mb']:.1f} MB")
    
    disk_info = disk_usage(".")
    if disk_info:
        logger.info(f"Disk usage: {disk_info['used_percent']:.1f}%")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {APP_NAME}")


# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "Endpoint not found",
            "error": f"The requested endpoint {request.url.path} was not found",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """Custom 500 handler."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with service information."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{APP_NAME}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; color: #333; border-bottom: 2px solid #007acc; padding-bottom: 20px; }}
            .section {{ margin: 20px 0; }}
            .phase {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #007acc; margin: 10px 0; }}
            .links {{ text-align: center; margin-top: 30px; }}
            .links a {{ display: inline-block; margin: 0 10px; padding: 10px 20px; background: #007acc; color: white; text-decoration: none; border-radius: 5px; }}
            .links a:hover {{ background: #005a9e; }}
            .status {{ color: #28a745; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛡️ {APP_NAME}</h1>
                <p>Version {APP_VERSION} - <span class="status">Running</span></p>
            </div>
            
            <div class="section">
                <h2>Two-Phase IoT Security Framework</h2>
                <p>A comprehensive solution for securing heterogeneous IoT networks through 
                device profiling and collaborative intrusion detection.</p>
                
                <div class="phase">
                    <h3>📊 Phase 1: Network Discovery & Device Profiling</h3>
                    <p>Extract features from PCAP files, classify IoT vs Non-IoT devices, 
                    and identify specific device types using hybrid feature extraction.</p>
                </div>
                
                <div class="phase">
                    <h3>🤖 Phase 2: Self-Labeled Federated Learning IDS</h3>
                    <p>Collaborative intrusion detection using MAML meta-learning, 
                    CT-AE privacy-preserving encoding, and BS-Agg helper selection.</p>
                </div>
            </div>
            
            <div class="section">
                <h3>🚀 Quick Start</h3>
                <p><strong>Phase 1:</strong> Upload PCAP → Extract Features → Train Models → Identify Devices</p>
                <p><strong>Phase 2:</strong> Setup Federation → Encode Data → Run Self-Labeling → Detect Intrusions</p>
            </div>
            
            <div class="links">
                <a href="/docs">📚 API Documentation</a>
                <a href="/redoc">📖 ReDoc</a>
                <a href="/health">💚 Health Check</a>
                <a href="/system/info">ℹ️ System Info</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Health check endpoint
@app.get("/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Basic system checks
        mem_info = memory_usage()
        disk_info = disk_usage(".")
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": APP_VERSION,
            "device": str(get_device()),
            "memory_mb": mem_info.get('rss_mb', 0) if mem_info else 0,
            "disk_usage_percent": disk_info.get('used_percent', 0) if disk_info else 0
        }
        
        return APIResponse(
            success=True,
            message="Service is healthy",
            data=health_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return APIResponse(
            success=False,
            message="Service health check failed",
            error=str(e)
        )


# System information endpoint
@app.get("/system/info", response_model=APIResponse)
async def system_info():
    """Get detailed system information."""
    try:
        import torch
        import platform
        import psutil
        
        system_info = {
            "service": {
                "name": APP_NAME,
                "version": APP_VERSION,
                "uptime": "N/A"  # Could implement uptime tracking
            },
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "architecture": platform.architecture()[0]
            },
            "hardware": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "device": str(get_device()),
                "cuda_available": torch.cuda.is_available()
            },
            "dependencies": {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
            }
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            system_info["hardware"]["gpu_name"] = torch.cuda.get_device_name()
            system_info["hardware"]["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return APIResponse(
            success=True,
            message="System information retrieved",
            data=system_info
        )
        
    except Exception as e:
        logger.error(f"System info failed: {e}")
        return APIResponse(
            success=False,
            message="Failed to retrieve system information",
            error=str(e)
        )


# Mount Phase routers
app.include_router(phase1_router)
app.include_router(phase2_router)


# Additional utility endpoints
@app.get("/config/validate")
async def validate_config(config_path: str = "config/default.yaml"):
    """Validate configuration file."""
    try:
        from src.common.io import load_config
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise HTTPException(status_code=404, detail=f"Configuration file not found: {config_path}")
        
        config = load_config(config_file)
        
        # Basic validation
        required_sections = ['paths', 'phase1', 'phase2']
        missing_sections = [section for section in required_sections if section not in config]
        
        validation_result = {
            "config_path": str(config_file),
            "valid": len(missing_sections) == 0,
            "sections": list(config.keys()),
            "missing_sections": missing_sections,
            "size_bytes": config_file.stat().st_size
        }
        
        return APIResponse(
            success=True,
            message="Configuration validation completed",
            data=validation_result
        )
        
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/status")
async def data_status():
    """Check status of data directories."""
    try:
        data_paths = {
            "raw": Path("data/raw"),
            "interim": Path("data/interim"), 
            "processed": Path("data/processed"),
            "models": Path("data/models")
        }
        
        status = {}
        for name, path in data_paths.items():
            if path.exists():
                files = list(path.glob("*"))
                status[name] = {
                    "exists": True,
                    "path": str(path),
                    "file_count": len(files),
                    "size_mb": sum(f.stat().st_size for f in files if f.is_file()) / (1024*1024)
                }
            else:
                status[name] = {
                    "exists": False,
                    "path": str(path),
                    "file_count": 0,
                    "size_mb": 0
                }
        
        return APIResponse(
            success=True,
            message="Data directory status retrieved",
            data=status
        )
        
    except Exception as e:
        logger.error(f"Data status check failed: {e}")
        return APIResponse(
            success=False,
            message="Failed to check data status",
            error=str(e)
        )


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


def main():
    """Main function to run the service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="IoT Security Framework API Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    logger.info(f"Starting {APP_NAME} on {args.host}:{args.port}")
    
    # Run the server
    uvicorn.run(
        "apps.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()
