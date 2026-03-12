"""
FastAPI endpoints for Phase 2 federated learning operations.

Provides REST API endpoints for:
- CT-AE encoding and similarity computation
- Federated learning coordination
- Self-labeling workflow management
- BS-Agg helper selection and aggregation
- Model adaptation and intrusion detection
"""

from typing import Dict, List, Optional, Any
import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
import torch
import numpy as np
import pandas as pd

from .federation.server import FederatedServer, ServerConfig
from .federation.client import FederatedClient, ClientConfig
from .federation.data_pipe import DataPipeline
from .models.cnn_1d import CNN1DClassifier
from ..common.schemas import (
    APIResponse, EncodingRequest, AggregationRequest, 
    AdaptationRequest, PredictionRequest
)
from ..common.logging import get_logger
from ..common.io import load_config, save_data, load_data
from ..common.utils import get_device

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/phase2", tags=["Phase 2 - Federated Learning IDS"])

# Global state for server and clients
_server: Optional[FederatedServer] = None
_clients: Dict[str, FederatedClient] = {}
_device = get_device()


def get_server() -> Optional[FederatedServer]:
    """Get federated server instance."""
    return _server


def get_client(client_id: str) -> Optional[FederatedClient]:
    """Get specific client instance."""
    return _clients.get(client_id)


@router.post("/setup/server", response_model=APIResponse)
async def setup_server(
    config_path: str = "config/phase2_federation.yaml",
    model_config: Optional[Dict] = None
):
    """
    Initialize federated server.
    
    Args:
        config_path: Path to federation configuration
        model_config: Optional model configuration override
        
    Returns:
        API response with server setup status
    """
    global _server
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Server configuration
        server_config = ServerConfig(**config.get('federation', {}))
        
        # Model configuration
        if model_config is None:
            model_config = config.get('cnn', {})
        
        # Create server
        _server = FederatedServer(model_config, server_config, _device)
        
        logger.info("Federated server initialized")
        
        return APIResponse(
            success=True,
            message="Federated server initialized successfully",
            data={
                "server_config": server_config.__dict__,
                "model_config": model_config,
                "device": str(_device)
            }
        )
        
    except Exception as e:
        logger.error(f"Server setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/setup/clients", response_model=APIResponse)
async def setup_clients(
    num_clients: int = 5,
    data_dir: str = "data/processed/phase2_local/",
    config_path: str = "config/phase2_federation.yaml"
):
    """
    Initialize federated clients with local data.
    
    Args:
        num_clients: Number of clients to create
        data_dir: Directory containing client data
        config_path: Path to configuration file
        
    Returns:
        API response with client setup status
    """
    global _clients
    
    try:
        # Load configuration
        config = load_config(config_path)
        model_config = config.get('cnn', {})
        
        # Initialize data pipeline
        pipeline = DataPipeline()
        
        # Load or create client data
        data_dir_path = Path(data_dir)
        if data_dir_path.exists():
            client_data = pipeline.load_client_data(data_dir_path)
        else:
            logger.warning(f"Data directory {data_dir} not found, creating synthetic data")
            # Create synthetic data for demonstration
            client_data = _create_synthetic_client_data(num_clients, pipeline)
        
        # Create clients
        _clients = {}
        for i, (client_id, data) in enumerate(list(client_data.items())[:num_clients]):
            client_config = ClientConfig(client_id=client_id, **config.get('client', {}))
            client = FederatedClient(client_id, model_config, client_config, _device)
            client.load_local_data(data)
            _clients[client_id] = client
            
            # Register with server if available
            if _server is not None:
                client_info = {
                    'num_samples': data.get('num_samples', 0),
                    'device_types': data.get('device_types', [])
                }
                _server.register_client(client_id, client_info)
        
        logger.info(f"Initialized {len(_clients)} federated clients")
        
        return APIResponse(
            success=True,
            message=f"Initialized {len(_clients)} federated clients",
            data={
                "num_clients": len(_clients),
                "client_ids": list(_clients.keys()),
                "data_dir": str(data_dir_path)
            }
        )
        
    except Exception as e:
        logger.error(f"Client setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prepare-data", response_model=APIResponse)
async def prepare_federated_data(
    profiles_path: str,
    output_dir: str = "data/processed/phase2_local/",
    num_clients: int = 10,
    heterogeneity: float = 0.7,
    attack_ratio: float = 0.3,
    background_tasks: BackgroundTasks = None
):
    """
    Prepare federated datasets from Phase 1 profiles.
    
    Args:
        profiles_path: Path to Phase 1 profiles
        output_dir: Output directory for client data
        num_clients: Number of federated clients
        heterogeneity: Statistical heterogeneity level
        attack_ratio: Fraction of attack samples
        background_tasks: FastAPI background tasks
        
    Returns:
        API response with data preparation status
    """
    try:
        # Start data preparation in background
        background_tasks.add_task(
            _prepare_data_background,
            profiles_path, output_dir, num_clients, heterogeneity, attack_ratio
        )
        
        return APIResponse(
            success=True,
            message="Data preparation started",
            data={
                "profiles_path": profiles_path,
                "output_dir": output_dir,
                "num_clients": num_clients,
                "heterogeneity": heterogeneity,
                "attack_ratio": attack_ratio
            }
        )
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _prepare_data_background(profiles_path: str, output_dir: str, 
                                 num_clients: int, heterogeneity: float, attack_ratio: float):
    """Background task for data preparation."""
    try:
        pipeline = DataPipeline()
        
        # Load Phase 1 profiles
        features_df, metadata_df = pipeline.load_phase1_data(profiles_path)
        
        # Simulate attacks
        features_attack, metadata_attack = pipeline.simulate_attacks(
            features_df, metadata_df, attack_ratio=attack_ratio
        )
        
        # Create federated splits
        client_data = pipeline.create_federated_splits(
            features_attack, metadata_attack, 
            num_clients=num_clients, 
            heterogeneity=heterogeneity
        )
        
        # Create meta-learning splits
        client_data = pipeline.create_meta_learning_splits(client_data)
        
        # Save client data
        pipeline.save_client_data(client_data, output_dir)
        
        logger.info(f"Data preparation completed: {len(client_data)} clients saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Background data preparation failed: {e}")


@router.post("/encode", response_model=APIResponse)
async def encode_client_data(request: EncodingRequest):
    """
    Encode client data using CT-AE.
    
    Args:
        request: Encoding request parameters
        
    Returns:
        API response with encoding results
    """
    try:
        client = get_client(request.client_id)
        if client is None:
            raise HTTPException(status_code=404, detail=f"Client {request.client_id} not found")
        
        # Train CT-AE if not already trained
        ct_ae_metrics = client.train_ct_ae()
        
        # Encode specified data type
        encodings = client.encode_data(request.data_type)
        
        # Update server with encodings
        if _server is not None:
            server_encodings = client.get_encodings_for_server()
            _server.update_client_encodings(request.client_id, server_encodings)
        
        return APIResponse(
            success=True,
            message=f"Encoded {request.data_type} data for client {request.client_id}",
            data={
                "client_id": request.client_id,
                "data_type": request.data_type,
                "encoding_shape": encodings.shape,
                "ct_ae_metrics": ct_ae_metrics
            }
        )
        
    except Exception as e:
        logger.error(f"Data encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similarity", response_model=APIResponse)
async def compute_similarity(client_id: str):
    """
    Compute similarity matrix for client.
    
    Args:
        client_id: Target client ID
        
    Returns:
        API response with similarity results
    """
    try:
        if _server is None:
            raise HTTPException(status_code=400, detail="Server not initialized")
        
        similarities = _server.compute_similarity_matrix(client_id)
        
        return APIResponse(
            success=True,
            message=f"Computed similarities for client {client_id}",
            data={
                "client_id": client_id,
                "similarities": similarities,
                "num_peers": len(similarities)
            }
        )
        
    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/aggregate", response_model=APIResponse)
async def similarity_based_aggregation(request: AggregationRequest):
    """
    Perform similarity-based aggregation (BS-Agg).
    
    Args:
        request: Aggregation request parameters
        
    Returns:
        API response with aggregation results
    """
    try:
        if _server is None:
            raise HTTPException(status_code=400, detail="Server not initialized")
        
        # Select helpers
        helpers = _server.select_helpers(request.client_id, request.gamma)
        
        if not helpers:
            raise HTTPException(
                status_code=400, 
                detail=f"No suitable helpers found for client {request.client_id}"
            )
        
        # Create annotation model
        annotation_model = _server.create_annotation_model(request.client_id, helpers)
        
        return APIResponse(
            success=True,
            message=f"Created annotation model for client {request.client_id}",
            data={
                "client_id": request.client_id,
                "selected_helpers": helpers,
                "num_helpers": len(helpers),
                "gamma": request.gamma,
                "model_created": True
            }
        )
        
    except Exception as e:
        logger.error(f"Similarity-based aggregation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adapt", response_model=APIResponse)
async def adapt_local_model(request: AdaptationRequest):
    """
    Adapt local model using self-labeling.
    
    Args:
        request: Adaptation request parameters
        
    Returns:
        API response with adaptation results
    """
    try:
        if _server is None:
            raise HTTPException(status_code=400, detail="Server not initialized")
        
        client = get_client(request.client_id)
        if client is None:
            raise HTTPException(status_code=404, detail=f"Client {request.client_id} not found")
        
        # Get annotation model from server
        helpers = _server.select_helpers(request.client_id)
        annotation_model = _server.create_annotation_model(request.client_id, helpers)
        
        # Perform self-labeling workflow
        workflow_results = client.self_labeling_workflow(annotation_model)
        
        return APIResponse(
            success=True,
            message=f"Completed self-labeling adaptation for client {request.client_id}",
            data={
                "client_id": request.client_id,
                "workflow_results": workflow_results,
                "adaptation_steps": request.adaptation_steps
            }
        )
        
    except Exception as e:
        logger.error(f"Model adaptation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=APIResponse)
async def predict_intrusions(request: PredictionRequest):
    """
    Predict intrusions using adapted model.
    
    Args:
        request: Prediction request parameters
        
    Returns:
        API response with prediction results
    """
    try:
        client = get_client(request.client_id)
        if client is None:
            raise HTTPException(status_code=404, detail=f"Client {request.client_id} not found")
        
        # Evaluate on query set
        metrics = client.evaluate_on_query()
        
        # Get detailed predictions if requested
        predictions_data = {}
        if request.return_probabilities:
            predictions, probabilities = client.model_trainer.predict(client.data_loaders['query'])
            predictions_data = {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist()
            }
        
        return APIResponse(
            success=True,
            message=f"Generated predictions for client {request.client_id}",
            data={
                "client_id": request.client_id,
                "metrics": metrics if request.return_metrics else {},
                **predictions_data
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/federated-round", response_model=APIResponse)
async def run_federated_round(round_number: Optional[int] = None):
    """
    Run one round of federated learning.
    
    Args:
        round_number: Optional round number
        
    Returns:
        API response with round results
    """
    try:
        if _server is None:
            raise HTTPException(status_code=400, detail="Server not initialized")
        
        if not _clients:
            raise HTTPException(status_code=400, detail="No clients available")
        
        # Select participating clients
        if round_number is None:
            round_number = _server.current_round + 1
        
        selected_clients = _server.select_clients(round_number)
        
        # Get global model
        global_model = _server.get_global_model()
        
        # Collect client updates
        client_updates = {}
        for client_id in selected_clients:
            if client_id in _clients:
                client_update = _clients[client_id].federated_round(global_model)
                client_updates[client_id] = client_update
        
        # Server aggregation
        round_results = _server.federated_round(client_updates)
        
        return APIResponse(
            success=True,
            message=f"Completed federated round {round_number}",
            data=round_results
        )
        
    except Exception as e:
        logger.error(f"Federated round failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-federation", response_model=APIResponse)
async def run_full_federation(
    num_rounds: int = 10,
    background_tasks: BackgroundTasks = None
):
    """
    Run complete federated learning process.
    
    Args:
        num_rounds: Number of federation rounds
        background_tasks: FastAPI background tasks
        
    Returns:
        API response with federation status
    """
    try:
        if _server is None or not _clients:
            raise HTTPException(status_code=400, detail="Server or clients not initialized")
        
        # Start federation in background
        background_tasks.add_task(_run_federation_background, num_rounds)
        
        return APIResponse(
            success=True,
            message=f"Started federated learning for {num_rounds} rounds",
            data={
                "num_rounds": num_rounds,
                "num_clients": len(_clients),
                "server_ready": True
            }
        )
        
    except Exception as e:
        logger.error(f"Federation startup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_federation_background(num_rounds: int):
    """Background task for running federation."""
    try:
        for round_num in range(1, num_rounds + 1):
            # Select clients
            selected_clients = _server.select_clients(round_num)
            global_model = _server.get_global_model()
            
            # Collect updates
            client_updates = {}
            for client_id in selected_clients:
                if client_id in _clients:
                    update = _clients[client_id].federated_round(global_model)
                    client_updates[client_id] = update
            
            # Server round
            _server.federated_round(client_updates)
            
            logger.info(f"Completed federation round {round_num}/{num_rounds}")
        
        logger.info(f"Federation completed after {num_rounds} rounds")
        
    except Exception as e:
        logger.error(f"Background federation failed: {e}")


@router.get("/status")
async def get_federation_status():
    """Get current federation status."""
    server_status = {}
    if _server is not None:
        server_status = _server.get_server_state()
    
    client_status = {}
    for client_id, client in _clients.items():
        client_status[client_id] = client.get_client_state()
    
    return APIResponse(
        success=True,
        message="Federation status retrieved",
        data={
            "server_initialized": _server is not None,
            "server_status": server_status,
            "num_clients": len(_clients),
            "client_status": client_status
        }
    )


@router.get("/summary")
async def get_federation_summary():
    """Get comprehensive federation summary."""
    if _server is None:
        raise HTTPException(status_code=400, detail="Server not initialized")
    
    summary = _server.get_federated_summary()
    
    return APIResponse(
        success=True,
        message="Federation summary retrieved",
        data=summary
    )


def _create_synthetic_client_data(num_clients: int, pipeline: DataPipeline) -> Dict[str, Dict]:
    """Create synthetic client data for demonstration."""
    logger.info("Creating synthetic client data for demonstration")
    
    # Generate synthetic profiles
    n_samples = 1000
    n_features = 35
    
    features_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    device_types = ['camera', 'sensor', 'switch', 'hub', 'thermostat']
    metadata_df = pd.DataFrame({
        'device_type': np.random.choice(device_types, n_samples),
        'is_iot': np.ones(n_samples, dtype=int)
    })
    
    # Simulate attacks
    features_attack, metadata_attack = pipeline.simulate_attacks(
        features_df, metadata_df, attack_ratio=0.3
    )
    
    # Create federated splits
    client_data = pipeline.create_federated_splits(
        features_attack, metadata_attack, 
        num_clients=num_clients, 
        heterogeneity=0.7
    )
    
    # Create meta-learning splits
    client_data = pipeline.create_meta_learning_splits(client_data)
    
    return client_data
