"""
FastAPI endpoints for Phase 1 profiling operations.

Provides REST API endpoints for:
- Feature extraction from PCAP files
- IoT device identification
- Device profiling and classification
- Model training and evaluation
"""

from typing import Dict, List, Optional, Union
import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import tempfile
import shutil

from .pcap_reader import FlowExtractor
from .feature_extractor import HybridFeatureExtractor
from .selectors import RandomForestSelector
from .train_identifiers import IoTClassifier, DeviceTypeClassifier, TwoStageClassifier
from .datasets import create_dataset_loader
from ..common.schemas import (
    APIResponse, ExtractionRequest, IdentificationRequest,
    FlowRecord, DeviceProfile
)
from ..common.logging import get_logger
from ..common.io import save_data, load_data, load_config
from ..common.utils import Timer

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/phase1", tags=["Phase 1 - Device Profiling"])

# Global state for loaded models and extractors
_feature_extractor = None
_feature_selector = None
_iot_classifier = None
_device_classifier = None
_two_stage_classifier = None


def get_feature_extractor() -> HybridFeatureExtractor:
    """Get or create feature extractor instance."""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = HybridFeatureExtractor()
    return _feature_extractor


def get_feature_selector() -> Optional[RandomForestSelector]:
    """Get feature selector if available."""
    return _feature_selector


def get_iot_classifier() -> Optional[IoTClassifier]:
    """Get IoT classifier if loaded."""
    return _iot_classifier


def get_device_classifier() -> Optional[DeviceTypeClassifier]:
    """Get device type classifier if loaded."""
    return _device_classifier


@router.post("/extract", response_model=APIResponse)
async def extract_features(request: ExtractionRequest, background_tasks: BackgroundTasks):
    """
    Extract features from PCAP file.
    
    Args:
        request: Extraction request parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        API response with extraction results
    """
    try:
        pcap_path = Path(request.pcap_path)
        if not pcap_path.exists():
            raise HTTPException(status_code=404, detail=f"PCAP file not found: {pcap_path}")
        
        logger.info(f"Starting feature extraction for {pcap_path}")
        
        # Determine output path
        if request.output_path:
            output_path = Path(request.output_path)
        else:
            output_path = pcap_path.with_suffix('.csv')
        
        # Extract flows and features in background
        background_tasks.add_task(
            _extract_features_background,
            pcap_path, output_path, request.window_size, request.flow_timeout
        )
        
        return APIResponse(
            success=True,
            message=f"Feature extraction started for {pcap_path.name}",
            data={
                "pcap_path": str(pcap_path),
                "output_path": str(output_path),
                "window_size": request.window_size,
                "flow_timeout": request.flow_timeout
            }
        )
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _extract_features_background(pcap_path: Path, output_path: Path,
                                     window_size: int, flow_timeout: int):
    """Background task for feature extraction."""
    try:
        with Timer(f"Feature extraction for {pcap_path.name}", logger):
            # Extract flows
            flow_extractor = FlowExtractor(
                flow_timeout=flow_timeout,
                window_size=window_size
            )
            flows = flow_extractor.extract_flows(pcap_path)
            
            # Extract features
            feature_extractor = get_feature_extractor()
            features_df = feature_extractor.extract_features_dataframe(flows)
            
            # Add flow metadata
            flow_df = flow_extractor.flows_to_dataframe(flows)
            
            # Combine flow info and features
            combined_df = pd.concat([flow_df, features_df], axis=1)
            
            # Save results
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_data(combined_df, output_path, format='csv')
            
            logger.info(f"Feature extraction completed: {len(combined_df)} flows, {output_path}")
            
    except Exception as e:
        logger.error(f"Background feature extraction failed: {e}")


@router.post("/extract/upload", response_model=APIResponse)
async def extract_features_upload(
    file: UploadFile = File(...),
    window_size: int = 60,
    flow_timeout: int = 120,
    background_tasks: BackgroundTasks = None
):
    """
    Extract features from uploaded PCAP file.
    
    Args:
        file: Uploaded PCAP file
        window_size: Time window size in seconds
        flow_timeout: Flow timeout in seconds
        background_tasks: FastAPI background tasks
        
    Returns:
        API response with extraction results
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pcap', '.pcapng')):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only .pcap and .pcapng files are supported."
            )
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pcap') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = Path(tmp_file.name)
        
        # Create output path
        output_path = tmp_path.with_suffix('.csv')
        
        # Start extraction
        background_tasks.add_task(
            _extract_features_background,
            tmp_path, output_path, window_size, flow_timeout
        )
        
        return APIResponse(
            success=True,
            message=f"Feature extraction started for uploaded file {file.filename}",
            data={
                "filename": file.filename,
                "output_path": str(output_path),
                "window_size": window_size,
                "flow_timeout": flow_timeout
            }
        )
        
    except Exception as e:
        logger.error(f"Upload feature extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/identify", response_model=APIResponse)
async def identify_devices(request: IdentificationRequest):
    """
    Identify IoT devices from extracted features.
    
    Args:
        request: Identification request parameters
        
    Returns:
        API response with identification results
    """
    try:
        csv_path = Path(request.csv_path)
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")
        
        logger.info(f"Starting device identification for {csv_path}")
        
        # Load features
        features_df = load_data(csv_path)
        
        # Get feature extractor to validate features
        feature_extractor = get_feature_extractor()
        feature_columns = feature_extractor.feature_names
        
        # Ensure all required features are present
        missing_features = set(feature_columns) - set(features_df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                features_df[feature] = 0.0
        
        # Select only feature columns
        X = features_df[feature_columns].values
        
        # Apply feature selection if available
        selector = get_feature_selector()
        if selector is not None:
            X = selector.transform(X)
        
        # Load models if specified
        iot_classifier = get_iot_classifier()
        device_classifier = get_device_classifier()
        
        if request.iotness_model_path:
            iot_classifier = IoTClassifier.load_model(request.iotness_model_path)
        
        if request.device_model_path:
            device_classifier = DeviceTypeClassifier.load_model(request.device_model_path)
        
        results = {}
        
        # IoT vs Non-IoT classification
        if iot_classifier is not None:
            iot_predictions = iot_classifier.predict(X)
            iot_probabilities = iot_classifier.predict_proba(X)
            
            results['iot_predictions'] = iot_predictions.tolist()
            results['iot_probabilities'] = iot_probabilities.tolist()
            results['iot_count'] = int(np.sum(iot_predictions))
            results['non_iot_count'] = int(len(iot_predictions) - np.sum(iot_predictions))
        
        # Device type classification for IoT devices
        if device_classifier is not None and 'iot_predictions' in results:
            iot_mask = np.array(iot_predictions) == 1
            if np.any(iot_mask):
                X_iot = X[iot_mask]
                device_predictions = device_classifier.predict(X_iot)
                device_probabilities = device_classifier.predict_proba(X_iot)
                
                # Create full predictions array
                full_device_predictions = np.full(len(X), 'Non-IoT', dtype=object)
                full_device_predictions[iot_mask] = device_predictions
                
                results['device_predictions'] = full_device_predictions.tolist()
                results['device_probabilities_iot'] = device_probabilities.tolist()
                results['device_type_counts'] = dict(zip(*np.unique(device_predictions, return_counts=True)))
        
        # Save results
        results_path = csv_path.with_suffix('.results.json')
        save_data(results, results_path, format='json')
        
        return APIResponse(
            success=True,
            message=f"Device identification completed for {csv_path.name}",
            data={
                "input_file": str(csv_path),
                "results_file": str(results_path),
                "total_flows": len(features_df),
                **results
            }
        )
        
    except Exception as e:
        logger.error(f"Device identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/iot-classifier", response_model=APIResponse)
async def train_iot_classifier(
    dataset_name: str = "iot_sentinel",
    data_dir: str = "data/raw",
    model_output_path: str = "data/models/iot_classifier.joblib",
    test_size: float = 0.2,
    background_tasks: BackgroundTasks = None
):
    """
    Train IoT vs Non-IoT classifier.
    
    Args:
        dataset_name: Name of dataset to use
        data_dir: Directory containing dataset files
        model_output_path: Path to save trained model
        test_size: Fraction of data for testing
        background_tasks: FastAPI background tasks
        
    Returns:
        API response with training results
    """
    try:
        logger.info(f"Starting IoT classifier training with {dataset_name}")
        
        # Start training in background
        background_tasks.add_task(
            _train_iot_classifier_background,
            dataset_name, data_dir, model_output_path, test_size
        )
        
        return APIResponse(
            success=True,
            message=f"IoT classifier training started with {dataset_name}",
            data={
                "dataset_name": dataset_name,
                "data_dir": data_dir,
                "model_output_path": model_output_path,
                "test_size": test_size
            }
        )
        
    except Exception as e:
        logger.error(f"IoT classifier training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _train_iot_classifier_background(dataset_name: str, data_dir: str,
                                         model_output_path: str, test_size: float):
    """Background task for IoT classifier training."""
    global _iot_classifier, _feature_selector
    
    try:
        with Timer(f"IoT classifier training", logger):
            # Load dataset
            loader = create_dataset_loader(dataset_name, data_dir)
            features_df, labels_df = loader.load()
            
            # Extract features using feature extractor
            feature_extractor = get_feature_extractor()
            
            # Ensure features match expected format
            expected_features = feature_extractor.feature_names
            available_features = [col for col in expected_features if col in features_df.columns]
            
            X = features_df[available_features].values
            y = labels_df['is_iot'].values
            
            # Feature selection
            selector = RandomForestSelector(n_features=35)
            selector.fit(X, y, available_features)
            X_selected = selector.transform(X)
            
            # Train classifier
            classifier = IoTClassifier()
            classifier.fit(X_selected, y, selector.get_selected_feature_names())
            
            # Evaluate
            metrics = classifier.cross_validate(X_selected, y, cv=5)
            
            # Save model and selector
            Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
            classifier.save_model(model_output_path)
            
            selector_path = Path(model_output_path).with_suffix('.selector.joblib')
            save_model(selector, selector_path)
            
            # Update global state
            _iot_classifier = classifier
            _feature_selector = selector
            
            logger.info(f"IoT classifier training completed: {metrics}")
            
    except Exception as e:
        logger.error(f"Background IoT classifier training failed: {e}")


@router.post("/train/device-classifier", response_model=APIResponse)
async def train_device_classifier(
    dataset_name: str = "iot_sentinel",
    data_dir: str = "data/raw",
    model_output_path: str = "data/models/device_classifier.joblib",
    test_size: float = 0.2,
    background_tasks: BackgroundTasks = None
):
    """
    Train device type classifier.
    
    Args:
        dataset_name: Name of dataset to use
        data_dir: Directory containing dataset files
        model_output_path: Path to save trained model
        test_size: Fraction of data for testing
        background_tasks: FastAPI background tasks
        
    Returns:
        API response with training results
    """
    try:
        logger.info(f"Starting device classifier training with {dataset_name}")
        
        # Start training in background
        background_tasks.add_task(
            _train_device_classifier_background,
            dataset_name, data_dir, model_output_path, test_size
        )
        
        return APIResponse(
            success=True,
            message=f"Device classifier training started with {dataset_name}",
            data={
                "dataset_name": dataset_name,
                "data_dir": data_dir,
                "model_output_path": model_output_path,
                "test_size": test_size
            }
        )
        
    except Exception as e:
        logger.error(f"Device classifier training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _train_device_classifier_background(dataset_name: str, data_dir: str,
                                            model_output_path: str, test_size: float):
    """Background task for device classifier training."""
    global _device_classifier
    
    try:
        with Timer(f"Device classifier training", logger):
            # Load dataset
            loader = create_dataset_loader(dataset_name, data_dir)
            features_df, labels_df = loader.load()
            
            # Filter only IoT devices
            iot_mask = labels_df['is_iot'] == 1
            features_iot = features_df[iot_mask]
            labels_iot = labels_df[iot_mask]
            
            # Extract features
            feature_extractor = get_feature_extractor()
            expected_features = feature_extractor.feature_names
            available_features = [col for col in expected_features if col in features_iot.columns]
            
            X = features_iot[available_features].values
            y = labels_iot['device_type'].values
            
            # Apply feature selection if available
            selector = get_feature_selector()
            if selector is not None:
                X = selector.transform(X)
                feature_names = selector.get_selected_feature_names()
            else:
                feature_names = available_features
            
            # Train classifier
            classifier = DeviceTypeClassifier()
            classifier.fit(X, y, feature_names)
            
            # Evaluate
            metrics = classifier.cross_validate(X, y, cv=5)
            
            # Save model
            Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
            classifier.save_model(model_output_path)
            
            # Update global state
            _device_classifier = classifier
            
            logger.info(f"Device classifier training completed: {metrics}")
            
    except Exception as e:
        logger.error(f"Background device classifier training failed: {e}")


@router.get("/models/load")
async def load_models(
    iot_model_path: Optional[str] = None,
    device_model_path: Optional[str] = None,
    selector_path: Optional[str] = None
):
    """
    Load trained models.
    
    Args:
        iot_model_path: Path to IoT classifier model
        device_model_path: Path to device type classifier model
        selector_path: Path to feature selector model
        
    Returns:
        API response with loading results
    """
    global _iot_classifier, _device_classifier, _feature_selector
    
    try:
        loaded_models = []
        
        if iot_model_path and Path(iot_model_path).exists():
            _iot_classifier = IoTClassifier.load_model(iot_model_path)
            loaded_models.append("IoT classifier")
        
        if device_model_path and Path(device_model_path).exists():
            _device_classifier = DeviceTypeClassifier.load_model(device_model_path)
            loaded_models.append("Device type classifier")
        
        if selector_path and Path(selector_path).exists():
            _feature_selector = load_model(selector_path)
            loaded_models.append("Feature selector")
        
        return APIResponse(
            success=True,
            message=f"Loaded models: {', '.join(loaded_models)}",
            data={
                "loaded_models": loaded_models,
                "iot_model_loaded": _iot_classifier is not None,
                "device_model_loaded": _device_classifier is not None,
                "selector_loaded": _feature_selector is not None
            }
        )
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status():
    """Get current status of Phase 1 components."""
    return APIResponse(
        success=True,
        message="Phase 1 status retrieved",
        data={
            "feature_extractor_loaded": _feature_extractor is not None,
            "feature_selector_loaded": _feature_selector is not None,
            "iot_classifier_loaded": _iot_classifier is not None,
            "device_classifier_loaded": _device_classifier is not None,
            "num_features": len(_feature_extractor.feature_names) if _feature_extractor else 0,
            "selected_features": len(_feature_selector.get_selected_features()) if _feature_selector else 0
        }
    )


@router.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """
    Download generated files (CSV, models, etc.).
    
    Args:
        file_path: Path to file to download
        
    Returns:
        File response
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"File download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
