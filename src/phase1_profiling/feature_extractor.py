"""
Hybrid Feature Extraction for IoT Device Profiling.

Implements the 58-feature hybrid set from Safi et al., combining:
- Size-related features (packet lengths, bytes)
- Time-related features (durations, inter-arrival times)
- Protocol-based features (TCP flags, protocol presence)
- Service-related features (port analysis, service detection)
- Statistical features (counts, ratios, distributions)
- DNS-specific features (query types, counts)

This comprehensive feature set enables robust IoT device identification
and behavioral profiling for the network discovery phase.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from collections import Counter
import socket
import struct

from .pcap_reader import NetworkFlow
from ..common.logging import get_logger
from ..common.utils import safe_divide

logger = get_logger(__name__)


class HybridFeatureExtractor:
    """
    Extract hybrid feature set from network flows.
    
    Implements the 58-feature hybrid approach from Safi et al. for
    comprehensive IoT device behavioral profiling.
    """
    
    def __init__(self):
        self.logger = logger
        self.feature_names = self._get_feature_names()
        
        # Well-known port ranges
        self.well_known_ports = set(range(1, 1024))
        self.registered_ports = set(range(1024, 49152))
        self.dynamic_ports = set(range(49152, 65536))
        
        # Service port mappings
        self.service_ports = {
            'http': [80, 8080, 8000, 8888],
            'https': [443, 8443],
            'dns': [53],
            'ntp': [123],
            'dhcp': [67, 68],
            'ssh': [22],
            'ftp': [21, 20],
            'telnet': [23],
            'smtp': [25, 587],
            'pop3': [110, 995],
            'imap': [143, 993],
            'snmp': [161, 162]
        }
    
    def extract_features(self, flows: Union[NetworkFlow, List[NetworkFlow]]) -> Union[Dict, List[Dict]]:
        """
        Extract features from network flows.
        
        Args:
            flows: Single flow or list of flows
            
        Returns:
            Feature dictionary or list of feature dictionaries
        """
        if isinstance(flows, NetworkFlow):
            return self._extract_single_flow_features(flows)
        else:
            return [self._extract_single_flow_features(flow) for flow in flows]
    
    def extract_features_dataframe(self, flows: List[NetworkFlow]) -> pd.DataFrame:
        """
        Extract features and return as DataFrame.
        
        Args:
            flows: List of network flows
            
        Returns:
            DataFrame with extracted features
        """
        feature_dicts = self.extract_features(flows)
        df = pd.DataFrame(feature_dicts)
        
        # Ensure all expected features are present
        for feature_name in self.feature_names:
            if feature_name not in df.columns:
                df[feature_name] = 0.0
        
        # Reorder columns to match expected feature order
        df = df[self.feature_names]
        
        return df
    
    def _extract_single_flow_features(self, flow: NetworkFlow) -> Dict[str, float]:
        """Extract all 58 features from a single flow."""
        features = {}
        
        # Size-related features (8 features)
        features.update(self._extract_size_features(flow))
        
        # Time-related features (13 features)  
        features.update(self._extract_time_features(flow))
        
        # Protocol-related features (12 features)
        features.update(self._extract_protocol_features(flow))
        
        # Service-related features (10 features)
        features.update(self._extract_service_features(flow))
        
        # Statistical features (13 features)
        features.update(self._extract_statistical_features(flow))
        
        # DNS-specific features (6 features)
        features.update(self._extract_dns_features(flow))
        
        return features
    
    def _extract_size_features(self, flow: NetworkFlow) -> Dict[str, float]:
        """Extract packet size and byte-related features."""
        all_lengths = flow.packet_lengths_forward + flow.packet_lengths_backward
        
        if not all_lengths:
            return {
                'packet_length_min': 0.0,
                'packet_length_max': 0.0,
                'packet_length_mean': 0.0,
                'packet_length_std': 0.0,
                'packet_length_variance': 0.0,
                'total_bytes_forward': 0.0,
                'total_bytes_backward': 0.0,
                'bytes_per_second': 0.0
            }
        
        lengths_array = np.array(all_lengths)
        
        return {
            'packet_length_min': float(np.min(lengths_array)),
            'packet_length_max': float(np.max(lengths_array)),
            'packet_length_mean': float(np.mean(lengths_array)),
            'packet_length_std': float(np.std(lengths_array)),
            'packet_length_variance': float(np.var(lengths_array)),
            'total_bytes_forward': float(flow.bytes_forward),
            'total_bytes_backward': float(flow.bytes_backward),
            'bytes_per_second': safe_divide(flow.total_bytes, flow.duration, 0.0)
        }
    
    def _extract_time_features(self, flow: NetworkFlow) -> Dict[str, float]:
        """Extract time-related and inter-arrival time features."""
        features = {
            'flow_duration': flow.duration,
        }
        
        # Flow-level inter-arrival times
        all_iats = flow.iat_forward + flow.iat_backward
        if all_iats:
            iat_array = np.array(all_iats)
            features.update({
                'flow_iat_mean': float(np.mean(iat_array)),
                'flow_iat_std': float(np.std(iat_array)),
                'flow_iat_max': float(np.max(iat_array)),
                'flow_iat_min': float(np.min(iat_array))
            })
        else:
            features.update({
                'flow_iat_mean': 0.0,
                'flow_iat_std': 0.0,
                'flow_iat_max': 0.0,
                'flow_iat_min': 0.0
            })
        
        # Forward direction IATs
        if flow.iat_forward:
            forward_iat = np.array(flow.iat_forward)
            features.update({
                'forward_iat_mean': float(np.mean(forward_iat)),
                'forward_iat_std': float(np.std(forward_iat))
            })
        else:
            features.update({
                'forward_iat_mean': 0.0,
                'forward_iat_std': 0.0
            })
        
        # Backward direction IATs
        if flow.iat_backward:
            backward_iat = np.array(flow.iat_backward)
            features.update({
                'backward_iat_mean': float(np.mean(backward_iat)),
                'backward_iat_std': float(np.std(backward_iat))
            })
        else:
            features.update({
                'backward_iat_mean': 0.0,
                'backward_iat_std': 0.0
            })
        
        # Active/Idle time features
        if flow.active_periods:
            active_array = np.array(flow.active_periods)
            features.update({
                'active_mean': float(np.mean(active_array)),
                'active_std': float(np.std(active_array))
            })
        else:
            features.update({
                'active_mean': 0.0,
                'active_std': 0.0
            })
        
        if flow.idle_periods:
            idle_array = np.array(flow.idle_periods)
            features.update({
                'idle_mean': float(np.mean(idle_array)),
                'idle_std': float(np.std(idle_array))
            })
        else:
            features.update({
                'idle_mean': 0.0,
                'idle_std': 0.0
            })
        
        return features
    
    def _extract_protocol_features(self, flow: NetworkFlow) -> Dict[str, float]:
        """Extract protocol-related and TCP flag features."""
        features = {
            # TCP flags
            'tcp_flag_count': sum(flow.tcp_flags.values()),
            'syn_flag_count': flow.tcp_flags.get('syn', 0),
            'ack_flag_count': flow.tcp_flags.get('ack', 0),
            'fin_flag_count': flow.tcp_flags.get('fin', 0),
            'rst_flag_count': flow.tcp_flags.get('rst', 0),
            'psh_flag_count': flow.tcp_flags.get('psh', 0),
            'urg_flag_count': flow.tcp_flags.get('urg', 0),
            
            # Protocol type encoding
            'protocol_type': self._encode_protocol(flow.flow_tuple.protocol),
            
            # Protocol presence flags
            'has_tcp': 1.0 if flow.flow_tuple.protocol == 'TCP' else 0.0,
            'has_udp': 1.0 if flow.flow_tuple.protocol == 'UDP' else 0.0,
            'has_icmp': 1.0 if flow.flow_tuple.protocol == 'ICMP' else 0.0,
            'has_arp': 0.0  # ARP flows are typically filtered out at IP level
        }
        
        return features
    
    def _extract_service_features(self, flow: NetworkFlow) -> Dict[str, float]:
        """Extract service and port-related features."""
        src_port = flow.flow_tuple.src_port
        dst_port = flow.flow_tuple.dst_port
        
        features = {
            'src_port': float(src_port),
            'dst_port': float(dst_port),
            'is_well_known_port': 1.0 if (src_port in self.well_known_ports or 
                                         dst_port in self.well_known_ports) else 0.0,
            'port_class_src': self._classify_port(src_port),
            'port_class_dst': self._classify_port(dst_port)
        }
        
        # Service detection based on ports
        for service, ports in self.service_ports.items():
            service_key = f'service_{service}'
            features[service_key] = 1.0 if (src_port in ports or dst_port in ports) else 0.0
        
        # Fill missing service features
        expected_services = ['http', 'https', 'dns', 'ntp', 'dhcp']
        for service in expected_services:
            service_key = f'service_{service}'
            if service_key not in features:
                features[service_key] = 0.0
        
        return features
    
    def _extract_statistical_features(self, flow: NetworkFlow) -> Dict[str, float]:
        """Extract statistical and ratio features."""
        features = {
            'packet_count_forward': float(flow.packet_count_forward),
            'packet_count_backward': float(flow.packet_count_backward),
            'packet_count_total': float(flow.total_packets),
            'packets_per_second': safe_divide(flow.total_packets, flow.duration, 0.0),
            'down_up_ratio': safe_divide(flow.packet_count_backward, 
                                       flow.packet_count_forward, 0.0)
        }
        
        # Average packet size
        if flow.total_packets > 0:
            features['avg_packet_size'] = flow.total_bytes / flow.total_packets
        else:
            features['avg_packet_size'] = 0.0
        
        # Packet size variance
        all_lengths = flow.packet_lengths_forward + flow.packet_lengths_backward
        if all_lengths:
            features['variance_packet_size'] = float(np.var(all_lengths))
        else:
            features['variance_packet_size'] = 0.0
        
        # Flow bytes per packet
        features['flow_bytes_per_packet'] = features['avg_packet_size']
        
        # Minimum segment size forward
        if flow.packet_lengths_forward:
            features['min_seg_size_forward'] = float(min(flow.packet_lengths_forward))
        else:
            features['min_seg_size_forward'] = 0.0
        
        # Subflow features (using entire flow as subflow for simplicity)
        features.update({
            'subflow_bytes_forward': float(flow.bytes_forward),
            'subflow_bytes_backward': float(flow.bytes_backward),
            'subflow_packets_forward': float(flow.packet_count_forward),
            'subflow_packets_backward': float(flow.packet_count_backward)
        })
        
        return features
    
    def _extract_dns_features(self, flow: NetworkFlow) -> Dict[str, float]:
        """Extract DNS-specific features."""
        features = {
            'dns_query_count': float(flow.dns_queries),
            'dns_response_count': float(flow.dns_responses),
            'dns_query_type_a': 0.0,  # Would need deeper packet inspection
            'dns_query_type_aaaa': 0.0,  # Would need deeper packet inspection
            'dns_query_type_ptr': 0.0,  # Would need deeper packet inspection
            'unique_dns_queries': float(min(flow.dns_queries, 1))  # Simplified
        }
        
        return features
    
    def _encode_protocol(self, protocol: str) -> float:
        """Encode protocol as numeric value."""
        protocol_map = {
            'TCP': 1.0,
            'UDP': 2.0,
            'ICMP': 3.0,
            'OTHER': 0.0
        }
        return protocol_map.get(protocol, 0.0)
    
    def _classify_port(self, port: int) -> float:
        """Classify port into categories."""
        if port in self.well_known_ports:
            return 1.0  # Well-known
        elif port in self.registered_ports:
            return 2.0  # Registered
        elif port in self.dynamic_ports:
            return 3.0  # Dynamic/Private
        else:
            return 0.0  # Unknown
    
    def _get_feature_names(self) -> List[str]:
        """Get ordered list of all 58 feature names."""
        return [
            # Size features (8)
            'packet_length_min', 'packet_length_max', 'packet_length_mean',
            'packet_length_std', 'packet_length_variance', 'total_bytes_forward',
            'total_bytes_backward', 'bytes_per_second',
            
            # Time features (13)
            'flow_duration', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max',
            'flow_iat_min', 'forward_iat_mean', 'forward_iat_std',
            'backward_iat_mean', 'backward_iat_std', 'active_mean', 'active_std',
            'idle_mean', 'idle_std',
            
            # Protocol features (12)
            'tcp_flag_count', 'syn_flag_count', 'ack_flag_count', 'fin_flag_count',
            'rst_flag_count', 'psh_flag_count', 'urg_flag_count', 'protocol_type',
            'has_tcp', 'has_udp', 'has_icmp', 'has_arp',
            
            # Service features (10)
            'src_port', 'dst_port', 'is_well_known_port', 'port_class_src',
            'port_class_dst', 'service_http', 'service_https', 'service_dns',
            'service_ntp', 'service_dhcp',
            
            # Statistical features (13)
            'packet_count_forward', 'packet_count_backward', 'packet_count_total',
            'packets_per_second', 'down_up_ratio', 'avg_packet_size',
            'variance_packet_size', 'flow_bytes_per_packet', 'min_seg_size_forward',
            'subflow_bytes_forward', 'subflow_bytes_backward',
            'subflow_packets_forward', 'subflow_packets_backward',
            
            # DNS features (6)
            'dns_query_count', 'dns_response_count', 'dns_query_type_a',
            'dns_query_type_aaaa', 'dns_query_type_ptr', 'unique_dns_queries'
        ]
    
    @property
    def num_features(self) -> int:
        """Get total number of features."""
        return len(self.feature_names)
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Get features grouped by category."""
        return {
            'size': self.feature_names[0:8],
            'time': self.feature_names[8:21],
            'protocol': self.feature_names[21:33],
            'service': self.feature_names[33:43],
            'statistical': self.feature_names[43:56],
            'dns': self.feature_names[56:62] if len(self.feature_names) >= 62 else self.feature_names[56:]
        }
    
    def validate_features(self, features: Union[Dict, pd.DataFrame]) -> bool:
        """
        Validate that all expected features are present.
        
        Args:
            features: Feature dictionary or DataFrame
            
        Returns:
            True if valid, False otherwise
        """
        if isinstance(features, dict):
            feature_keys = set(features.keys())
        elif isinstance(features, pd.DataFrame):
            feature_keys = set(features.columns)
        else:
            return False
        
        expected_features = set(self.feature_names)
        missing_features = expected_features - feature_keys
        
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
            return False
        
        return True
    
    def normalize_features(self, features_df: pd.DataFrame, 
                          method: str = 'minmax') -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features using specified method.
        
        Args:
            features_df: DataFrame with features
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            Tuple of (normalized_df, normalization_params)
        """
        from ..common.utils import normalize_features
        
        feature_values = features_df[self.feature_names].values
        normalized_values, params = normalize_features(feature_values, method)
        
        normalized_df = features_df.copy()
        normalized_df[self.feature_names] = normalized_values
        
        return normalized_df, params
