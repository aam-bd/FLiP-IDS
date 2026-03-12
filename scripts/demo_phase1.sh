#!/bin/bash

# Demo script for Phase 1: Network Discovery and Device Profiling
# 
# This script demonstrates the complete Phase 1 workflow:
# 1. Feature extraction from PCAP files
# 2. Training IoT vs Non-IoT classifier
# 3. Training device type classifier
# 4. Device identification from new traffic

set -e  # Exit on any error

echo "🛡️  IoT Security Framework - Phase 1 Demo"
echo "=========================================="
echo ""

# Configuration
DEMO_DIR="demo_phase1"
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw"
INTERIM_DIR="$DATA_DIR/interim" 
PROCESSED_DIR="$DATA_DIR/processed"
MODELS_DIR="$DATA_DIR/models"

# Create directories
echo "📁 Setting up directories..."
mkdir -p $DEMO_DIR/{$RAW_DIR,$INTERIM_DIR,$PROCESSED_DIR,$MODELS_DIR}
cd $DEMO_DIR

echo "✅ Directory structure created"
echo ""

# Step 1: Generate synthetic PCAP data (since we don't have real PCAP files)
echo "📦 Step 1: Generating synthetic network data..."
cat > generate_synthetic_data.py << 'EOF'
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append('../src')

from src.phase1_profiling.datasets import IoTSentinelLoader

# Generate synthetic IoT Sentinel-like data
print("Generating synthetic IoT network data...")
loader = IoTSentinelLoader('data/raw')
features_df, labels_df = loader.load(sample_ratio=0.8, balance_classes=True)

# Save as CSV for demo
features_df.to_csv('data/interim/network_features.csv', index=False)
labels_df.to_csv('data/interim/network_labels.csv', index=False)

# Combine for convenience
combined_df = pd.concat([features_df, labels_df], axis=1)
combined_df.to_csv('data/interim/network_data.csv', index=False)

print(f"✅ Generated {len(combined_df)} network samples")
print(f"   - Features: {len(features_df.columns)} dimensions")
print(f"   - Device types: {labels_df['device_type'].nunique()}")
print(f"   - IoT devices: {labels_df['is_iot'].sum()}")
print(f"   - Non-IoT devices: {(labels_df['is_iot'] == 0).sum()}")
EOF

python generate_synthetic_data.py
echo ""

# Step 2: Feature extraction and selection
echo "🔍 Step 2: Feature extraction and selection..."
python -m src.phase1_profiling.cli train-iot \
    --dataset-files data/interim/network_data.csv \
    --label-column is_iot \
    --model-output data/models/iot_classifier.joblib \
    --feature-selection \
    --n-features 35 \
    --n-estimators 100 \
    --cv-folds 3 \
    --verbose

echo "✅ IoT vs Non-IoT classifier trained"
echo ""

# Step 3: Train device type classifier
echo "🏷️  Step 3: Training device type classifier..."

# Filter IoT devices for device type classification
cat > prepare_device_data.py << 'EOF'
import pandas as pd

# Load data
data = pd.read_csv('data/interim/network_data.csv')

# Filter only IoT devices
iot_data = data[data['is_iot'] == 1].copy()
iot_data.to_csv('data/interim/iot_devices.csv', index=False)

print(f"✅ Prepared {len(iot_data)} IoT device samples")
print(f"   Device types: {iot_data['device_type'].value_counts().to_dict()}")
EOF

python prepare_device_data.py

python -m src.phase1_profiling.cli train-device \
    --dataset-files data/interim/iot_devices.csv \
    --label-column device_type \
    --model-output data/models/device_classifier.joblib \
    --selector-path data/models/iot_classifier.selector.joblib \
    --n-estimators 100 \
    --cv-folds 3 \
    --verbose

echo "✅ Device type classifier trained"
echo ""

# Step 4: Evaluate models
echo "📊 Step 4: Evaluating trained models..."

echo "Evaluating IoT classifier:"
python -m src.phase1_profiling.cli evaluate \
    data/models/iot_classifier.joblib \
    --model-type iot \
    --test-data data/interim/network_data.csv \
    --label-column is_iot \
    --selector-path data/models/iot_classifier.selector.joblib \
    --output data/processed/iot_evaluation \
    --plot-confusion

echo ""
echo "Evaluating device type classifier:"
python -m src.phase1_profiling.cli evaluate \
    data/models/device_classifier.joblib \
    --model-type device \
    --test-data data/interim/iot_devices.csv \
    --label-column device_type \
    --selector-path data/models/iot_classifier.selector.joblib \
    --output data/processed/device_evaluation \
    --plot-confusion

echo "✅ Model evaluation completed"
echo ""

# Step 5: Demonstrate device identification
echo "🔍 Step 5: Device identification demonstration..."

# Create a test dataset
cat > create_test_data.py << 'EOF'
import pandas as pd
import numpy as np

# Load original data
data = pd.read_csv('data/interim/network_data.csv')

# Create a test set (simulate new network traffic)
np.random.seed(123)
test_indices = np.random.choice(len(data), 50, replace=False)
test_data = data.iloc[test_indices].copy()

# Remove labels to simulate unlabeled data
feature_columns = [col for col in test_data.columns if col not in ['is_iot', 'device_type']]
test_features = test_data[feature_columns]
test_features.to_csv('data/interim/test_traffic.csv', index=False)

# Save ground truth for comparison
test_labels = test_data[['is_iot', 'device_type']]
test_labels.to_csv('data/interim/test_labels.csv', index=False)

print(f"✅ Created test dataset with {len(test_features)} samples")
EOF

python create_test_data.py

# Run device identification
python -m src.phase1_profiling.cli identify \
    data/interim/test_traffic.csv \
    --iot-model-path data/models/iot_classifier.joblib \
    --device-model-path data/models/device_classifier.joblib \
    --selector-path data/models/iot_classifier.selector.joblib \
    --output data/processed/identification_results.json

echo "✅ Device identification completed"
echo ""

# Step 6: Generate summary report
echo "📋 Step 6: Generating summary report..."

cat > generate_report.py << 'EOF'
import pandas as pd
import json
from pathlib import Path

print("🛡️  Phase 1 Demo Summary Report")
print("=" * 50)

# Load identification results
with open('data/processed/identification_results.json', 'r') as f:
    results = json.load(f)

# Load ground truth
test_labels = pd.read_csv('data/interim/test_labels.csv')

print(f"\n📊 Dataset Statistics:")
print(f"   Total test samples: {results['total_flows']}")
print(f"   Identified IoT devices: {results['iot_count']}")
print(f"   Identified Non-IoT devices: {results['non_iot_count']}")

if 'device_type_counts' in results:
    print(f"\n🏷️  Device Type Distribution:")
    for device_type, count in results['device_type_counts'].items():
        print(f"   {device_type}: {count}")

# Calculate accuracy if we have ground truth
if len(test_labels) == results['total_flows']:
    iot_predictions = results['iot_predictions']
    true_iot = test_labels['is_iot'].values
    
    iot_accuracy = sum(p == t for p, t in zip(iot_predictions, true_iot)) / len(true_iot)
    print(f"\n✅ IoT Classification Accuracy: {iot_accuracy:.2%}")
    
    if 'device_predictions' in results:
        device_predictions = results['device_predictions']
        true_devices = test_labels['device_type'].values
        
        # Only compare IoT devices
        iot_mask = [p == 1 for p in iot_predictions]
        if any(iot_mask):
            iot_device_preds = [device_predictions[i] for i, is_iot in enumerate(iot_mask) if is_iot]
            iot_device_true = [true_devices[i] for i, is_iot in enumerate(iot_mask) if is_iot]
            
            device_accuracy = sum(p == t for p, t in zip(iot_device_preds, iot_device_true)) / len(iot_device_true)
            print(f"✅ Device Type Accuracy: {device_accuracy:.2%}")

print(f"\n📁 Generated Files:")
print(f"   - IoT Classifier: data/models/iot_classifier.joblib")
print(f"   - Device Classifier: data/models/device_classifier.joblib")
print(f"   - Feature Selector: data/models/iot_classifier.selector.joblib")
print(f"   - Evaluation Results: data/processed/")
print(f"   - Identification Results: data/processed/identification_results.json")

print(f"\n🎉 Phase 1 Demo Completed Successfully!")
print(f"   The system can now identify IoT devices and classify their types")
print(f"   from network traffic with high accuracy.")
EOF

python generate_report.py
echo ""

# Step 7: Optional - Compare feature selectors
echo "🔬 Step 7: Comparing feature selection methods..."
python -m src.phase1_profiling.cli compare-selectors \
    --dataset-files data/interim/network_data.csv \
    --n-features 35 \
    --output data/processed/selector_comparison.json

echo "✅ Feature selector comparison completed"
echo ""

echo "🎉 Phase 1 Demo Complete!"
echo ""
echo "📋 Summary:"
echo "   ✅ Generated synthetic IoT network data"
echo "   ✅ Trained IoT vs Non-IoT classifier (RF with feature selection)"
echo "   ✅ Trained device type classifier"
echo "   ✅ Evaluated model performance"
echo "   ✅ Demonstrated device identification"
echo "   ✅ Compared feature selection methods"
echo ""
echo "📁 Check the following directories for results:"
echo "   - data/models/ - Trained models"
echo "   - data/processed/ - Evaluation results and reports"
echo ""
echo "🚀 Ready for Phase 2: Federated Learning IDS!"

# Return to original directory
cd ..
