"""
PCAP reading and flow extraction utilities.

Implements passive network traffic capture parsing and flow-based feature extraction
using both dpkt and scapy for robust PCAP file handling. Extracts flows based on
5-tuple (src_ip, dst_ip, src_port, dst_port, protocol) with bidirectional tracking.
"""

import struct
import socket
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator
import dpkt
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether, ARP
from scapy.layers.dns import DNS
import numpy as np
import pandas as pd

from ..common.logging import get_logger, ProgressLogger
from ..common.utils import Timer

logger = get_logger(__name__)

# Flow record structure
FlowTuple = namedtuple('FlowTuple', ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'])

class NetworkFlow:
    """Represents a network flow with extracted features."""
    
    def __init__(self, flow_tuple: FlowTuple, first_packet_time: float):
        self.flow_tuple = flow_tuple
        self.start_time = first_packet_time
        self.last_time = first_packet_time
        
        # Packet and byte counters
        self.packet_count_forward = 0
        self.packet_count_backward = 0
        self.bytes_forward = 0
        self.bytes_backward = 0
        
        # Packet sizes
        self.packet_lengths_forward = []
        self.packet_lengths_backward = []
        
        # Inter-arrival times
        self.iat_forward = []
        self.iat_backward = []
        self.last_forward_time = None
        self.last_backward_time = None
        
        # TCP flags
        self.tcp_flags = defaultdict(int)
        
        # Protocol specific features
        self.dns_queries = 0
        self.dns_responses = 0
        self.ntp_requests = 0
        self.http_requests = 0
        
        # Active/Idle periods
        self.active_periods = []
        self.idle_periods = []
        self.last_activity_time = first_packet_time
        
    def add_packet(self, packet_size: int, timestamp: float, is_forward: bool, 
                   tcp_flags: Optional[int] = None, protocol_info: Optional[Dict] = None):
        """Add a packet to this flow."""
        self.last_time = max(self.last_time, timestamp)
        
        if is_forward:
            self.packet_count_forward += 1
            self.bytes_forward += packet_size
            self.packet_lengths_forward.append(packet_size)
            
            if self.last_forward_time is not None:
                iat = timestamp - self.last_forward_time
                self.iat_forward.append(iat)
            self.last_forward_time = timestamp
        else:
            self.packet_count_backward += 1
            self.bytes_backward += packet_size
            self.packet_lengths_backward.append(packet_size)
            
            if self.last_backward_time is not None:
                iat = timestamp - self.last_backward_time
                self.iat_backward.append(iat)
            self.last_backward_time = timestamp
        
        # TCP flags
        if tcp_flags is not None:
            if tcp_flags & dpkt.tcp.TH_SYN:
                self.tcp_flags['syn'] += 1
            if tcp_flags & dpkt.tcp.TH_ACK:
                self.tcp_flags['ack'] += 1
            if tcp_flags & dpkt.tcp.TH_FIN:
                self.tcp_flags['fin'] += 1
            if tcp_flags & dpkt.tcp.TH_RST:
                self.tcp_flags['rst'] += 1
            if tcp_flags & dpkt.tcp.TH_PUSH:
                self.tcp_flags['psh'] += 1
            if tcp_flags & dpkt.tcp.TH_URG:
                self.tcp_flags['urg'] += 1
        
        # Protocol specific information
        if protocol_info:
            if 'dns_query' in protocol_info:
                self.dns_queries += 1
            if 'dns_response' in protocol_info:
                self.dns_responses += 1
            if 'ntp_request' in protocol_info:
                self.ntp_requests += 1
            if 'http_request' in protocol_info:
                self.http_requests += 1
        
        # Update activity tracking
        if timestamp - self.last_activity_time > 1.0:  # 1 second idle threshold
            idle_period = timestamp - self.last_activity_time
            self.idle_periods.append(idle_period)
        
        self.last_activity_time = timestamp
    
    @property
    def duration(self) -> float:
        """Flow duration in seconds."""
        return max(0.0, self.last_time - self.start_time)
    
    @property
    def total_packets(self) -> int:
        """Total packet count."""
        return self.packet_count_forward + self.packet_count_backward
    
    @property
    def total_bytes(self) -> int:
        """Total byte count."""
        return self.bytes_forward + self.bytes_backward
    
    def get_flow_id(self) -> str:
        """Generate unique flow identifier."""
        return f"{self.flow_tuple.src_ip}:{self.flow_tuple.src_port}->" \
               f"{self.flow_tuple.dst_ip}:{self.flow_tuple.dst_port}_{self.flow_tuple.protocol}"


class PcapReader:
    """PCAP file reader with multiple backend support."""
    
    def __init__(self, use_scapy: bool = False):
        """
        Initialize PCAP reader.
        
        Args:
            use_scapy: Whether to use scapy instead of dpkt (slower but more robust)
        """
        self.use_scapy = use_scapy
        self.logger = logger
    
    def read_pcap(self, pcap_path: Union[str, Path]) -> Iterator[Tuple[float, bytes]]:
        """
        Read packets from PCAP file.
        
        Args:
            pcap_path: Path to PCAP file
            
        Yields:
            Tuples of (timestamp, packet_data)
        """
        pcap_path = Path(pcap_path)
        
        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")
        
        self.logger.info(f"Reading PCAP file: {pcap_path}")
        
        if self.use_scapy:
            yield from self._read_with_scapy(pcap_path)
        else:
            yield from self._read_with_dpkt(pcap_path)
    
    def _read_with_dpkt(self, pcap_path: Path) -> Iterator[Tuple[float, bytes]]:
        """Read PCAP using dpkt (faster)."""
        try:
            with open(pcap_path, 'rb') as f:
                pcap_reader = dpkt.pcap.Reader(f)
                
                for timestamp, packet_data in pcap_reader:
                    yield timestamp, packet_data
                    
        except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, struct.error) as e:
            self.logger.warning(f"dpkt failed to read {pcap_path}: {e}, falling back to scapy")
            yield from self._read_with_scapy(pcap_path)
    
    def _read_with_scapy(self, pcap_path: Path) -> Iterator[Tuple[float, bytes]]:
        """Read PCAP using scapy (more robust)."""
        try:
            packets = scapy.rdpcap(str(pcap_path))
            
            for packet in packets:
                if hasattr(packet, 'time'):
                    timestamp = float(packet.time)
                else:
                    timestamp = 0.0
                
                packet_data = bytes(packet)
                yield timestamp, packet_data
                
        except Exception as e:
            self.logger.error(f"Failed to read PCAP file {pcap_path}: {e}")
            raise


class FlowExtractor:
    """Extract network flows from PCAP data."""
    
    def __init__(self, flow_timeout: int = 120, window_size: int = 60):
        """
        Initialize flow extractor.
        
        Args:
            flow_timeout: Flow timeout in seconds
            window_size: Time window size for flow aggregation in seconds
        """
        self.flow_timeout = flow_timeout
        self.window_size = window_size
        self.flows = {}  # Dict[FlowTuple, NetworkFlow]
        self.logger = logger
    
    def extract_flows(self, pcap_path: Union[str, Path]) -> List[NetworkFlow]:
        """
        Extract flows from PCAP file.
        
        Args:
            pcap_path: Path to PCAP file
            
        Returns:
            List of NetworkFlow objects
        """
        pcap_reader = PcapReader()
        flows = {}
        packet_count = 0
        
        with Timer("Flow extraction", self.logger):
            for timestamp, packet_data in pcap_reader.read_pcap(pcap_path):
                packet_count += 1
                
                if packet_count % 10000 == 0:
                    self.logger.info(f"Processed {packet_count} packets, {len(flows)} flows")
                
                flow_info = self._parse_packet(packet_data, timestamp)
                if flow_info is None:
                    continue
                
                flow_tuple, packet_size, is_forward, tcp_flags, protocol_info = flow_info
                
                # Get or create flow
                if flow_tuple not in flows:
                    flows[flow_tuple] = NetworkFlow(flow_tuple, timestamp)
                
                flows[flow_tuple].add_packet(
                    packet_size, timestamp, is_forward, tcp_flags, protocol_info
                )
                
                # Clean up old flows
                if packet_count % 50000 == 0:
                    flows = self._cleanup_old_flows(flows, timestamp)
        
        self.logger.info(f"Extracted {len(flows)} flows from {packet_count} packets")
        return list(flows.values())
    
    def _parse_packet(self, packet_data: bytes, timestamp: float) -> Optional[Tuple]:
        """Parse packet and extract flow information."""
        try:
            # Try dpkt first
            eth = dpkt.ethernet.Ethernet(packet_data)
            
            if not isinstance(eth.data, dpkt.ip.IP):
                return None
            
            ip = eth.data
            src_ip = socket.inet_ntoa(ip.src)
            dst_ip = socket.inet_ntoa(ip.dst)
            
            protocol_info = {}
            tcp_flags = None
            src_port = dst_port = 0
            protocol = 'OTHER'
            
            if isinstance(ip.data, dpkt.tcp.TCP):
                tcp = ip.data
                src_port = tcp.sport
                dst_port = tcp.dport
                protocol = 'TCP'
                tcp_flags = tcp.flags
                
                # Check for HTTP
                if src_port == 80 or dst_port == 80 or src_port == 8080 or dst_port == 8080:
                    protocol_info['http_request'] = True
                
            elif isinstance(ip.data, dpkt.udp.UDP):
                udp = ip.data
                src_port = udp.sport
                dst_port = udp.dport
                protocol = 'UDP'
                
                # Check for DNS
                if src_port == 53 or dst_port == 53:
                    try:
                        dns = dpkt.dns.DNS(udp.data)
                        if dns.qr == 0:  # Query
                            protocol_info['dns_query'] = True
                        else:  # Response
                            protocol_info['dns_response'] = True
                    except:
                        pass
                
                # Check for NTP
                elif src_port == 123 or dst_port == 123:
                    protocol_info['ntp_request'] = True
                    
            elif isinstance(ip.data, dpkt.icmp.ICMP):
                protocol = 'ICMP'
            
            # Create flow tuple (normalize direction)
            if (src_ip, src_port) < (dst_ip, dst_port):
                flow_tuple = FlowTuple(src_ip, dst_ip, src_port, dst_port, protocol)
                is_forward = True
            else:
                flow_tuple = FlowTuple(dst_ip, src_ip, dst_port, src_port, protocol)
                is_forward = False
            
            packet_size = len(packet_data)
            
            return flow_tuple, packet_size, is_forward, tcp_flags, protocol_info
            
        except Exception as e:
            # Fallback to scapy parsing
            try:
                packet = scapy.Ether(packet_data)
                return self._parse_packet_scapy(packet, timestamp)
            except:
                return None
    
    def _parse_packet_scapy(self, packet, timestamp: float) -> Optional[Tuple]:
        """Parse packet using scapy (fallback method)."""
        try:
            if not packet.haslayer(IP):
                return None
            
            ip_layer = packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            
            protocol_info = {}
            tcp_flags = None
            src_port = dst_port = 0
            protocol = 'OTHER'
            
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                src_port = tcp_layer.sport
                dst_port = tcp_layer.dport
                protocol = 'TCP'
                
                # Convert scapy flags to dpkt format
                flags = 0
                if tcp_layer.flags.S: flags |= dpkt.tcp.TH_SYN
                if tcp_layer.flags.A: flags |= dpkt.tcp.TH_ACK
                if tcp_layer.flags.F: flags |= dpkt.tcp.TH_FIN
                if tcp_layer.flags.R: flags |= dpkt.tcp.TH_RST
                if tcp_layer.flags.P: flags |= dpkt.tcp.TH_PUSH
                if tcp_layer.flags.U: flags |= dpkt.tcp.TH_URG
                tcp_flags = flags
                
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                src_port = udp_layer.sport
                dst_port = udp_layer.dport
                protocol = 'UDP'
                
                # Check for DNS
                if packet.haslayer(DNS):
                    dns_layer = packet[DNS]
                    if dns_layer.qr == 0:
                        protocol_info['dns_query'] = True
                    else:
                        protocol_info['dns_response'] = True
                
            elif packet.haslayer(ICMP):
                protocol = 'ICMP'
            
            # Create flow tuple (normalize direction)
            if (src_ip, src_port) < (dst_ip, dst_port):
                flow_tuple = FlowTuple(src_ip, dst_ip, src_port, dst_port, protocol)
                is_forward = True
            else:
                flow_tuple = FlowTuple(dst_ip, src_ip, dst_port, src_port, protocol)
                is_forward = False
            
            packet_size = len(packet)
            
            return flow_tuple, packet_size, is_forward, tcp_flags, protocol_info
            
        except Exception as e:
            return None
    
    def _cleanup_old_flows(self, flows: Dict, current_time: float) -> Dict:
        """Remove flows that have timed out."""
        active_flows = {}
        timeout_threshold = current_time - self.flow_timeout
        
        for flow_tuple, flow in flows.items():
            if flow.last_time >= timeout_threshold:
                active_flows[flow_tuple] = flow
        
        return active_flows
    
    def flows_to_dataframe(self, flows: List[NetworkFlow]) -> pd.DataFrame:
        """
        Convert flows to pandas DataFrame for further processing.
        
        Args:
            flows: List of NetworkFlow objects
            
        Returns:
            DataFrame with basic flow information
        """
        records = []
        
        for flow in flows:
            record = {
                'flow_id': flow.get_flow_id(),
                'src_ip': flow.flow_tuple.src_ip,
                'dst_ip': flow.flow_tuple.dst_ip,
                'src_port': flow.flow_tuple.src_port,
                'dst_port': flow.flow_tuple.dst_port,
                'protocol': flow.flow_tuple.protocol,
                'start_time': datetime.fromtimestamp(flow.start_time),
                'end_time': datetime.fromtimestamp(flow.last_time),
                'duration': flow.duration,
                'total_packets': flow.total_packets,
                'total_bytes': flow.total_bytes,
                'packets_forward': flow.packet_count_forward,
                'packets_backward': flow.packet_count_backward,
                'bytes_forward': flow.bytes_forward,
                'bytes_backward': flow.bytes_backward,
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def extract_flows_windowed(self, pcap_path: Union[str, Path], 
                             window_size: Optional[int] = None) -> List[List[NetworkFlow]]:
        """
        Extract flows in time windows.
        
        Args:
            pcap_path: Path to PCAP file
            window_size: Window size in seconds (uses instance default if None)
            
        Returns:
            List of flow lists, one per time window
        """
        if window_size is None:
            window_size = self.window_size
        
        all_flows = self.extract_flows(pcap_path)
        
        if not all_flows:
            return []
        
        # Group flows by time windows
        start_time = min(flow.start_time for flow in all_flows)
        end_time = max(flow.last_time for flow in all_flows)
        
        windows = []
        current_window_start = start_time
        
        while current_window_start < end_time:
            current_window_end = current_window_start + window_size
            
            window_flows = [
                flow for flow in all_flows
                if flow.start_time >= current_window_start and flow.start_time < current_window_end
            ]
            
            if window_flows:
                windows.append(window_flows)
            
            current_window_start = current_window_end
        
        self.logger.info(f"Created {len(windows)} time windows of {window_size}s each")
        return windows
