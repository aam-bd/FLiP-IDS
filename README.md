# FLiP-IDS Framework: Two-Phase Network Profiling + Self-Labeled Federated Learning IDS

A production-ready Python framework implementing a two-phase approach for securing heterogeneous IoT networks:

1. **Phase 1**: Network discovery and device profiling using hybrid feature extraction
2. **Phase 2**: Self-labeled personalized federated learning (SOH-FL) for collaborative intrusion detection

### For full setup along with the datasets:
https://drive.google.com/drive/u/0/folders/17qrmjJ2zyU_td13x1xkz4hNNSojdcK_C

## 🏗️ Architecture Overview

### Phase 1: Network Discovery and Device Profiling
- **PCAP Processing**: Passive network traffic capture and flow extraction
- **Hybrid Feature Extraction**: 58-feature set from Safi et al. (size, time, protocol, service, statistical, DNS)
- **Feature Selection**: Random Forest-based importance selection (top-35 features)
- **Two-Stage Classification**: 
  1. IoT vs Non-IoT classification
  2. Device type identification for IoT devices

### Phase 2: Self-Labeled Federated Learning IDS (SOH-FL)
- **Meta-Learning**: MAML-based personalized federated learning
- **Privacy-Preserving Encoding**: Cosine-Targeted Autoencoder (CT-AE)
- **Similarity-Based Aggregation**: BS-Agg for helper selection
- **Self-Labeling**: Automatic annotation of unlabeled attack data
- **Collaborative Detection**: Zero-day attack detection without manual labeling

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/aam-bd/FLiP-IDS.git
cd iot-secure-framework

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Demo A: Phase 1 Device Profiling

```bash
# Extract features from PCAP file
python -m src.phase1_profiling.cli extract sample.pcap --output features.csv

# Train IoT classifier
python -m src.phase1_profiling.cli train-iot --dataset iot_sentinel --model-output models/iot_classifier.joblib

# Train device type classifier
python -m src.phase1_profiling.cli train-device --dataset iot_sentinel --model-output models/device_classifier.joblib

# Identify devices in new traffic
python -m src.phase1_profiling.cli identify features.csv --iot-model-path models/iot_classifier.joblib --device-model-path models/device_classifier.joblib
```

### Demo B: Phase 2 Federated IDS

```bash
# Prepare federated datasets from Phase 1 profiles
python -m src.phase2_ids.cli prepare-local --profiles data/processed/profiles.parquet --output data/processed/phase2_local/

# Run federated learning with self-labeling
python -m src.phase2_ids.cli run-federation --config config/phase2_federation.yaml

# Encode and aggregate for similarity-based helper selection
python -m src.phase2_ids.cli encode-and-aggregate --gamma 3
```

## 📊 Expected Performance

### Phase 1 Classification
- **IoT vs Non-IoT**: >90% accuracy with Random Forest
- **Device Type**: >85% accuracy across 18+ device types
- **Feature Reduction**: 58 → 35 features with <2% accuracy loss

### Phase 2 Federated IDS
- **Self-Labeling Accuracy**: >80% on zero-day attacks
- **Adaptation Speed**: 3-5 gradient steps for personalization
- **Privacy**: Only 32-dim latent vectors shared (vs 35-dim features)
- **Collaboration**: 15-25% improvement with top-3 helpers vs isolated learning

## 🔧 Configuration

### Phase 1 Configuration (`config/phase1_features.yaml`)
```yaml
feature_categories:
  size_features: [packet_length_min, packet_length_max, ...]
  time_features: [flow_duration, flow_iat_mean, ...]
  protocol_features: [tcp_flag_count, syn_flag_count, ...]
  # ... more categories

selection:
  method: random_forest_importance
  threshold_alpha: 0.003
  top_k: 35
```

### Phase 2 Configuration (`config/phase2_federation.yaml`)
```yaml
federation:
  total_rounds: 50
  num_clients: 10
  client_participation_rate: 0.6
  maml_inner_lr: 0.001  # α in paper
  maml_outer_lr: 0.005  # β in paper
  gamma_top_helpers: 3

ct_autoencoder:
  latent_dim: 32
  reconstruction_weight: 0.7
  cosine_similarity_weight: 0.3
```

## 🏃‍♂️ Running the Complete Pipeline

### 1. Start the FastAPI Service
```bash
python -m apps.service
# Access API at http://localhost:8000/docs
```

### 2. Phase 1: Device Profiling
```bash
# Upload PCAP and extract features
curl -X POST "http://localhost:8000/phase1/extract/upload" \
  -F "file=@sample.pcap"

# Train models
curl -X POST "http://localhost:8000/phase1/train/iot-classifier" \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "iot_sentinel"}'

# Identify devices
curl -X POST "http://localhost:8000/phase1/identify" \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "features.csv"}'
```

### 3. Phase 2: Federated IDS
```bash
# Encode client data
curl -X POST "http://localhost:8000/phase2/encode" \
  -H "Content-Type: application/json" \
  -d '{"client_id": "client_01"}'

# Run similarity-based aggregation
curl -X POST "http://localhost:8000/phase2/aggregate" \
  -H "Content-Type: application/json" \
  -d '{"client_id": "client_01", "gamma": 3}'

# Adapt and predict
curl -X POST "http://localhost:8000/phase2/adapt" \
  -H "Content-Type: application/json" \
  -d '{"client_id": "client_01"}'
```

## 📁 Project Structure

```
iot-secure-framework/
├── config/                     # Configuration files
│   ├── default.yaml
│   ├── phase1_features.yaml
│   └── phase2_federation.yaml
├── data/                       # Data directories
│   ├── raw/                   # Raw PCAP files
│   ├── interim/               # Intermediate processing
│   ├── processed/             # Final processed data
│   └── models/                # Trained models
├── src/
│   ├── common/                # Shared utilities
│   │   ├── io.py             # I/O operations
│   │   ├── logging.py        # Logging setup
│   │   ├── metrics.py        # Evaluation metrics
│   │   ├── utils.py          # General utilities
│   │   └── schemas.py        # Pydantic data models
│   ├── phase1_profiling/     # Phase 1 implementation
│   │   ├── pcap_reader.py    # PCAP parsing and flow extraction
│   │   ├── feature_extractor.py  # 58-feature hybrid extraction
│   │   ├── selectors.py      # Feature selection (RF importance)
│   │   ├── train_identifiers.py  # Two-stage classification
│   │   ├── datasets.py       # Dataset loaders
│   │   ├── api.py           # FastAPI endpoints
│   │   └── cli.py           # Command-line interface
│   └── phase2_ids/           # Phase 2 SOH-FL implementation
│       ├── models/
│       │   ├── cnn_1d.py    # 1D CNN for tabular data
│       │   ├── autoencoders.py  # CT-AE implementation
│       │   └── maml.py      # MAML meta-learning
│       ├── federation/
│       │   ├── server.py    # FedAvg + BS-Agg server
│       │   ├── client.py    # Gateway client simulation
│       │   └── data_pipe.py # Phase1→Phase2 data pipeline
│       ├── api.py           # FastAPI endpoints
│       └── cli.py           # Command-line interface
├── apps/
│   └── service.py            # Main FastAPI application
├── tests/                    # Unit and integration tests
├── scripts/                  # Demo and utility scripts
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_phase1_features.py    # Phase 1 feature extraction
pytest tests/test_phase2_ctae.py        # CT-AE autoencoder
pytest tests/test_phase2_maml.py        # MAML meta-learning
pytest tests/test_end_to_end.py         # Full pipeline test
```

## 📚 Key References

1. **Safi et al.** - "Hybrid Feature Set for IoT Device Identification" - Basis for the 58-feature extraction approach
2. **SOH-FL Paper** - "Stones from Other Hills: Federated Learning for IoT Intrusion Detection" - Self-labeled personalized federated learning methodology
3. **MAML** - Model-Agnostic Meta-Learning for fast adaptation
4. **Behavioral Monitoring in IoT** - Network profiling and device fingerprinting techniques

## 🛠️ Development

### Adding New Features
- **Phase 1**: Extend `HybridFeatureExtractor` with new feature categories
- **Phase 2**: Add new model architectures in `src/phase2_ids/models/`
- **Federation**: Implement new aggregation strategies in `federation/server.py`

### Custom Datasets
- Implement `DatasetLoader` subclass in `src/phase1_profiling/datasets.py`
- Follow the interface: `load() -> (features_df, labels_df)`

### Configuration
- Extend YAML configs for new parameters
- Use `src/common/io.load_config()` for parsing

## 📈 Performance Monitoring

The framework includes comprehensive metrics tracking:
- **Phase 1**: Accuracy, precision, recall, F1-score, confusion matrices
- **Phase 2**: Meta-learning loss, adaptation accuracy, similarity scores, federated aggregation metrics
- **Privacy**: Latent space dimensionality reduction, reconstruction quality

## 🔒 Security Considerations

- **Privacy**: Raw traffic stays local, only CT-AE encodings (32-dim) are shared
- **Robustness**: Statistical heterogeneity simulation for realistic federated scenarios  
- **Scalability**: Designed for 10-100 IoT gateways with thousands of devices each

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review example configurations in `config/`

---

**Note**: This framework implements cutting-edge research in IoT security and federated learning. For production deployment, ensure proper security auditing and compliance with relevant regulations.
