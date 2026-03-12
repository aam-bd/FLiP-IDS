"""
Phase 1: Network Discovery and Device Profiling

This module implements the first phase of the IoT security framework,
focusing on passive network discovery, hybrid feature extraction,
and IoT device identification and classification.

Key components:
- PCAP parsing and flow extraction
- Hybrid feature set extraction (58 features from Safi et al.)
- Feature selection using Random Forest importance
- Two-stage classification: IoT vs Non-IoT, then device type identification
"""

from .pcap_reader import PcapReader, FlowExtractor
from .feature_extractor import HybridFeatureExtractor
from .selectors import FeatureSelector, RandomForestSelector
from .train_identifiers import IoTClassifier, DeviceTypeClassifier
from .datasets import DatasetLoader, IoTSentinelLoader

__all__ = [
    "PcapReader",
    "FlowExtractor", 
    "HybridFeatureExtractor",
    "FeatureSelector",
    "RandomForestSelector",
    "IoTClassifier",
    "DeviceTypeClassifier",
    "DatasetLoader",
    "IoTSentinelLoader"
]
