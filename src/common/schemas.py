"""
Pydantic schemas for data contracts and API models.

Defines the data structures used throughout the IoT Security Framework
for type safety and validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import numpy as np


class FlowRecord(BaseModel):
    """Network flow record with extracted features."""
    
    flow_id: str = Field(..., description="Unique flow identifier")
    ts_start: datetime = Field(..., description="Flow start timestamp")
    ts_end: datetime = Field(..., description="Flow end timestamp") 
    src_ip: str = Field(..., description="Source IP address")
    dst_ip: str = Field(..., description="Destination IP address")
    src_port: int = Field(..., description="Source port")
    dst_port: int = Field(..., description="Destination port")
    protocol: str = Field(..., description="Protocol (TCP/UDP/ICMP)")
    
    # Feature vector (58 hybrid features)
    features: Dict[str, float] = Field(..., description="Extracted feature vector")
    
    # Classification results
    is_iot: Optional[bool] = Field(None, description="IoT vs Non-IoT classification")
    device_type: Optional[str] = Field(None, description="Specific device type")
    confidence_iot: Optional[float] = Field(None, description="IoT classification confidence")
    confidence_device: Optional[float] = Field(None, description="Device type confidence")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DeviceProfile(BaseModel):
    """Device behavioral profile aggregated from flow records."""
    
    device_id: str = Field(..., description="Unique device identifier")
    device_type: str = Field(..., description="Device type classification")
    mac_address: Optional[str] = Field(None, description="MAC address if available")
    ip_addresses: List[str] = Field(default_factory=list, description="Associated IP addresses")
    
    # Behavioral statistics
    total_flows: int = Field(..., description="Total number of flows observed")
    observation_period: float = Field(..., description="Observation period in hours")
    avg_bytes_per_flow: float = Field(..., description="Average bytes per flow")
    avg_packets_per_flow: float = Field(..., description="Average packets per flow")
    protocol_distribution: Dict[str, float] = Field(..., description="Protocol usage distribution")
    
    # Feature statistics for normal behavior baseline
    feature_means: Dict[str, float] = Field(..., description="Mean values for each feature")
    feature_stds: Dict[str, float] = Field(..., description="Standard deviations for each feature")
    
    # Temporal patterns
    active_hours: List[int] = Field(default_factory=list, description="Hours of typical activity")
    communication_patterns: Dict[str, Any] = Field(default_factory=dict, description="Communication patterns")
    
    created_at: datetime = Field(default_factory=datetime.now, description="Profile creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")


class ModelWeights(BaseModel):
    """Model weights for federated learning."""
    
    model_id: str = Field(..., description="Unique model identifier")
    client_id: Optional[str] = Field(None, description="Client ID if local model")
    weights: Dict[str, List[float]] = Field(..., description="Model parameters as serializable dict")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    @validator('weights')
    def validate_weights(cls, v):
        """Ensure weights are serializable."""
        # Convert numpy arrays to lists if needed
        for key, value in v.items():
            if hasattr(value, 'tolist'):
                v[key] = value.tolist()
        return v


class ClientState(BaseModel):
    """State of a federated learning client."""
    
    client_id: str = Field(..., description="Unique client identifier")
    round_number: int = Field(..., description="Current federation round")
    
    # Data information
    num_samples: int = Field(..., description="Number of local training samples")
    device_types: List[str] = Field(..., description="Device types handled by this client")
    
    # Training state
    local_epochs_completed: int = Field(0, description="Local epochs completed this round")
    local_loss: Optional[float] = Field(None, description="Local training loss")
    local_accuracy: Optional[float] = Field(None, description="Local validation accuracy")
    
    # CT-AE encoded vectors
    historical_encoding: Optional[List[float]] = Field(None, description="CT-AE encoding of historical data")
    current_encoding: Optional[List[float]] = Field(None, description="CT-AE encoding of current unlabeled data")
    
    # Helper information for BS-Agg
    helper_similarities: Dict[str, float] = Field(default_factory=dict, description="Similarity scores with other clients")
    selected_helpers: List[str] = Field(default_factory=list, description="Selected helper client IDs")
    
    # Adaptation state
    prelabeled_samples: int = Field(0, description="Number of prelabeled samples")
    adaptation_loss: Optional[float] = Field(None, description="Loss after local adaptation")
    final_accuracy: Optional[float] = Field(None, description="Accuracy on test query set")
    
    last_updated: datetime = Field(default_factory=datetime.now, description="Last state update")


class FederationConfig(BaseModel):
    """Configuration for federated learning setup."""
    
    # Basic federation parameters
    num_clients: int = Field(10, description="Total number of clients")
    rounds: int = Field(50, description="Number of federation rounds")
    participation_rate: float = Field(0.6, description="Fraction of clients participating per round")
    local_epochs: int = Field(3, description="Local training epochs per round")
    
    # Optimization parameters
    learning_rate: float = Field(0.005, description="Learning rate")
    batch_size: int = Field(40, description="Batch size for training")
    
    # MAML parameters
    maml_inner_lr: float = Field(0.001, description="MAML inner loop learning rate (alpha)")
    maml_outer_lr: float = Field(0.005, description="MAML outer loop learning rate (beta)")
    maml_inner_steps: int = Field(1, description="MAML inner loop gradient steps")
    
    # Self-labeling parameters
    gamma_top_helpers: int = Field(3, description="Number of top helper clients for BS-Agg")
    similarity_threshold: float = Field(0.1, description="Minimum similarity for helper selection")
    adaptation_steps: int = Field(5, description="Local adaptation steps after prelabeling")
    
    # CT-AE parameters
    latent_dim: int = Field(32, description="CT-AE latent dimension")
    ct_ae_epochs: int = Field(10, description="CT-AE training epochs")
    reconstruction_weight: float = Field(0.7, description="Reconstruction loss weight")
    cosine_weight: float = Field(0.3, description="Cosine similarity loss weight")
    
    @validator('participation_rate')
    def validate_participation_rate(cls, v):
        if not 0 < v <= 1:
            raise ValueError('Participation rate must be between 0 and 1')
        return v
    
    @validator('gamma_top_helpers')
    def validate_gamma_helpers(cls, v, values):
        if 'num_clients' in values and v >= values['num_clients']:
            raise ValueError('gamma_top_helpers must be less than num_clients')
        return v


class ExtractionRequest(BaseModel):
    """Request for feature extraction from PCAP file."""
    
    pcap_path: str = Field(..., description="Path to PCAP file")
    output_path: Optional[str] = Field(None, description="Optional output path for CSV")
    window_size: int = Field(60, description="Time window size in seconds")
    flow_timeout: int = Field(120, description="Flow timeout in seconds")


class IdentificationRequest(BaseModel):
    """Request for device identification from extracted features."""
    
    csv_path: str = Field(..., description="Path to CSV file with extracted features")
    iotness_model_path: Optional[str] = Field(None, description="Path to IoT vs Non-IoT model")
    device_model_path: Optional[str] = Field(None, description="Path to device type model")


class EncodingRequest(BaseModel):
    """Request for CT-AE encoding of client data."""
    
    client_id: str = Field(..., description="Client identifier")
    data_type: str = Field(..., description="Type of data to encode: 'historical' or 'current'")


class AggregationRequest(BaseModel):
    """Request for similarity-based aggregation (BS-Agg)."""
    
    client_id: str = Field(..., description="Target client identifier")
    gamma: int = Field(3, description="Number of top helpers to select")
    similarity_threshold: float = Field(0.1, description="Minimum similarity threshold")


class AdaptationRequest(BaseModel):
    """Request for local model adaptation after prelabeling."""
    
    client_id: str = Field(..., description="Client identifier")
    adaptation_steps: int = Field(5, description="Number of adaptation steps")
    learning_rate: float = Field(0.01, description="Adaptation learning rate")


class PredictionRequest(BaseModel):
    """Request for intrusion detection prediction."""
    
    client_id: str = Field(..., description="Client identifier")
    return_probabilities: bool = Field(False, description="Return class probabilities")
    return_metrics: bool = Field(True, description="Return evaluation metrics")


class APIResponse(BaseModel):
    """Generic API response wrapper."""
    
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
