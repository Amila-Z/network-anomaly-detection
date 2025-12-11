#!/usr/bin/env python3
"""
Network Traffic Anomaly Detection System - UNSW-NB15 Dataset
This program detects suspicious network traffic patterns using real attack data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Network Traffic Anomaly Detection System".center(70))
print("UNSW-NB15 Dataset Analysis".center(70))
print("=" * 70)
print()

# Load the UNSW-NB15 dataset
print("[*] Loading UNSW-NB15 training dataset...")
print("    This may take 10-30 seconds depending on your system...")

try:
    # Load the dataset
    df = pd.read_csv('UNSW_NB15_training-set.csv')
    print(f"[+] Successfully loaded dataset!")
    print(f"    Total records: {len(df):,}")
    print(f"    Features: {len(df.columns)}")
    print()
    
    # Display first few column names to understand the structur
    print("[*] Dataset columns available:")
    print(f"    {', '.join(df.columns[:10].tolist())}...")
    print()
    
except FileNotFoundError:
    print("[!] ERROR: UNSW_NB15_training-set.csv not found!")
    print("    Please make sure the file is in the same directory as this script.")
    print("    Current directory:", os.getcwd())
    exit(1)
except Exception as e:
    print(f"[!] ERROR loading dataset: {e}")
    exit(1)

# Data preprocesing
print("[*] Preprocessing data...")

# For analysis, let's use a sample if dataset is very large (for faster processing)
# we can increase this number or remove sampling for full analysis
SAMPLE_SIZE = 50000  # Using 50K records for reasonable processing time
if len(df) > SAMPLE_SIZE:
    print(f"[*] Sampling {SAMPLE_SIZE:,} records for analysis (from {len(df):,} total)")
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
else:
    df_sample = df.copy()

traffic_data = df_sample.copy()

# Map protocol numbers to names for better readability
protocol_map = {0: 'HOPOPT', 1: 'ICMP', 6: 'TCP', 17: 'UDP', 47: 'GRE', 
                89: 'OSPF', 132: 'SCTP'}

# Identify key columns (UNSW-NB15 has specific column names)
# Common columns: srcip, sport, dstip, dsport, proto, state, dur, sbytes, dbytes, 
#                 sttl, dttl, sloss, dloss, service, Sload, Dload, etc.

print(f"[+] Working with {len(traffic_data):,} network packets")
print()

# ============================================================================
# DETECTION METHOD 1: RULE-BASED DETECTION
# ============================================================================

print("=" * 70)
print("[*] Running Rule-Based Anomaly Detection on Real Attack Data...")
print("=" * 70)

anomalies_found = 0

# Rule 1: Detect Suspicious Source IPs (high traffic volume)
print("\n[Rule 1] Checking for High-Volume Traffic Sources...")
if 'srcip' in traffic_data.columns:
    packet_threshold = 100
    ip_counts = traffic_data['srcip'].value_counts()
    
    high_traffic_ips = ip_counts[ip_counts > packet_threshold]
    if len(high_traffic_ips) > 0:
        print(f"  ⚠️  ALERT: Found {len(high_traffic_ips)} IPs with excessive traffic")
        for ip, count in high_traffic_ips.head(5).items():
            print(f"      IP {ip}: {count} packets (threshold: {packet_threshold})")
        print(f"      Severity: HIGH - Potential DoS Attack Pattern")
        anomalies_found += 1
    else:
        print(f"  ✓ No excessive traffic detected (threshold: {packet_threshold} packets/IP)")
else:
    print("  [!] Source IP column not found in dataset")

# Rule 2: Detect Port Scanning Behavior
print("\n[Rule 2] Checking for Port Scanning Activity...")
if 'sport' in traffic_data.columns and 'srcip' in traffic_data.columns:
    port_threshold = 20
    scan_detected = False
    
    for src_ip in traffic_data['srcip'].unique()[:100]:  # Check top 100 IPs
        src_data = traffic_data[traffic_data['srcip'] == src_ip]
        unique_ports = src_data['dsport'].nunique() if 'dsport' in traffic_data.columns else 0
        
        if unique_ports > port_threshold:
            if not scan_detected:
                print(f"  ⚠️  ALERT: Port scanning detected!")
                scan_detected = True
            print(f"      IP {src_ip}: accessed {unique_ports} unique ports")
    
    if scan_detected:
        print(f"      Severity: HIGH - Reconnaissance Activity")
        anomalies_found += 1
    else:
        print(f"  ✓ No port scanning detected (threshold: {port_threshold} ports/IP)")
else:
    print("  [!] Required columns not found for port scan detection")

# Rule 3: Detect Unusual Packet/Byte Sizes
print("\n[Rule 3] Checking for Unusual Data Transfer Sizes...")
if 'sbytes' in traffic_data.columns or 'dbytes' in traffic_data.columns:
    # Check for unusually large data transfers
    byte_col = 'sbytes' if 'sbytes' in traffic_data.columns else 'dbytes'
    
    if traffic_data[byte_col].max() > 0:
        byte_threshold = traffic_data[byte_col].quantile(0.95)  # 95th percentile
        large_transfers = traffic_data[traffic_data[byte_col] > byte_threshold]
        
        if len(large_transfers) > 0:
            print(f"  ⚠️  ALERT: Found {len(large_transfers)} unusually large transfers")
            print(f"      Average size: {large_transfers[byte_col].mean():.0f} bytes")
            print(f"      Threshold: {byte_threshold:.0f} bytes (95th percentile)")
            print(f"      Severity: MEDIUM - Possible Data Exfiltration")
            anomalies_found += 1
        else:
            print(f"  ✓ All data transfers within normal range")
    else:
        print("  [!] No byte data available")
else:
    print("  [!] Byte columns not found in dataset")

# Rule 4: Check Actual Attack Labels (Ground Truth)
print("\n[Rule 4] Verifying Against Known Attacks (Ground Truth)...")
if 'attack_cat' in traffic_data.columns or 'label' in traffic_data.columns:
    label_col = 'attack_cat' if 'attack_cat' in traffic_data.columns else 'label'
    
    # Count actual attacks in the dataset
    if label_col == 'attack_cat':
        attacks = traffic_data[traffic_data[label_col] != 'Normal']
        normal = traffic_data[traffic_data[label_col] == 'Normal']
    else:
        attacks = traffic_data[traffic_data[label_col] == 1]
        normal = traffic_data[traffic_data[label_col] == 0]
    
    print(f"  📊 Ground Truth Analysis:")
    print(f"      Normal traffic: {len(normal):,} packets ({len(normal)/len(traffic_data)*100:.1f}%)")
    print(f"      Attack traffic: {len(attacks):,} packets ({len(attacks)/len(traffic_data)*100:.1f}%)")
    
    if 'attack_cat' in traffic_data.columns:
        print(f"\n  🎯 Attack Types Present:")
        attack_types = attacks['attack_cat'].value_counts()
        for attack_type, count in attack_types.head(10).items():
            print(f"      {attack_type}: {count:,} packets")
    
    anomalies_found += 1
else:
    print("  [!] Attack label column not found")

print(f"\n[+] Rule-Based Detection Complete: Found {anomalies_found} anomaly categories")

# ============================================================================
# DETECTION METHOD 2: MACHINE LEARNING DETECTION
# ============================================================================

print("\n" + "=" * 70)
print("[*] Running Machine Learning Detection (Isolation Forest)...")
print("=" * 70)

# Select numerical features for ML
numerical_cols = traffic_data.select_dtypes(include=[np.number]).columns.tolist()

# Remove label columns if present
label_cols = ['label', 'attack_cat', 'id']
features = [col for col in numerical_cols if col not in label_cols]

print(f"[*] Using {len(features)} numerical features for ML detection")
print(f"    Sample features: {', '.join(features[:8])}...")

# Prepare data for ML
X = traffic_data[features].copy()

# Handle mising values and infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Limit features if too many (for computational efficiency)
if len(features) > 20:
    # Select most important features (those with highest variance)
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()].tolist()[:20]
    X = X[selected_features]
    print(f"[*] Selected top 20 features with highest variance")

# Normalize features
print("[*] Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
print("[*] Training Isolation Forest model...")
print("    This may take 30-60 seconds...")

iso_forest = IsolationForest(
    contamination=0.1,
    random_state=42,
    n_estimators=100,
    max_samples=min(1000, len(X_scaled)),  # Limit samples for speed
    n_jobs=-1  # Use all CPU cores
)

predictions = iso_forest.fit_predict(X_scaled)
anomaly_scores = iso_forest.score_samples(X_scaled)

# Add predictions
traffic_data['is_anomaly'] = predictions
traffic_data['anomaly_score'] = anomaly_scores

ml_anomalies = (predictions == -1).sum()
print(f"[+] ML Detection Complete: Found {ml_anomalies:,} anomalous packets ({ml_anomalies/len(traffic_data)*100:.1f}%)")

# Compare ML predictions with ground truth (if available)
if 'label' in traffic_data.columns or 'attack_cat' in traffic_data.columns:
    label_col = 'label' if 'label' in traffic_data.columns else 'attack_cat'
    
    if label_col == 'label':
        actual_attacks = (traffic_data[label_col] == 1).sum()
        true_positives = ((traffic_data['is_anomaly'] == -1) & (traffic_data[label_col] == 1)).sum()
        false_positives = ((traffic_data['is_anomaly'] == -1) & (traffic_data[label_col] == 0)).sum()
    else:
        actual_attacks = (traffic_data[label_col] != 'Normal').sum()
        true_positives = ((traffic_data['is_anomaly'] == -1) & (traffic_data[label_col] != 'Normal')).sum()
        false_positives = ((traffic_data['is_anomaly'] == -1) & (traffic_data[label_col] == 'Normal')).sum()
    
    accuracy = (true_positives / ml_anomalies * 100) if ml_anomalies > 0 else 0
    
    print(f"\n  📈 ML Performance Metrics:")
    print(f"      True Positives: {true_positives:,} (correctly identified attacks)")
    print(f"      False Positives: {false_positives:,} (false alarms)")
    print(f"      Detection Accuracy: {accuracy:.1f}%")

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 70)
print("[*] Generating Visualizations from Real Attack Data...")
print("=" * 70)

import os
output_dir = 'output_UNSW'
os.makedirs(output_dir, exist_ok=True)

# Visualization 1: Attack Types Distribution (if available)
print("[*] Creating visualization 1/4: Attack Types Distribution...")
if 'attack_cat' in traffic_data.columns:
    plt.figure(figsize=(12, 6))
    attack_dist = traffic_data['attack_cat'].value_counts().head(10)
    colors = plt.cm.Set3(range(len(attack_dist)))
    plt.bar(range(len(attack_dist)), attack_dist.values, color=colors)
    plt.xticks(range(len(attack_dist)), attack_dist.index, rotation=45, ha='right')
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Number of Packets', fontsize=12)
    plt.title('Distribution of Attack Types in UNSW-NB15 Dataset', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/attack_types_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir}/attack_types_distribution.png")
else:
    # Alternative: Source IP distribution
    plt.figure(figsize=(12, 6))
    top_ips = traffic_data['srcip'].value_counts().head(10) if 'srcip' in traffic_data.columns else pd.Series([1])
    plt.bar(range(len(top_ips)), top_ips.values, color='steelblue')
    plt.xticks(range(len(top_ips)), [f"IP_{i+1}" for i in range(len(top_ips))], rotation=45)
    plt.xlabel('Source IP', fontsize=12)
    plt.ylabel('Packet Count', fontsize=12)
    plt.title('Top Source IPs by Traffic Volume', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/attack_types_distribution.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_dir}/attack_types_distribution.png")

# Visualization 2: Data Transfer Size Distribution
print("[*] Creating visualization 2/4: Data Transfer Size Distribution...")
plt.figure(figsize=(10, 6))
byte_col = 'sbytes' if 'sbytes' in traffic_data.columns else 'dbytes' if 'dbytes' in traffic_data.columns else None
if byte_col and traffic_data[byte_col].max() > 0:
    # Remove zeros and outliers for better visualization
    data_for_hist = traffic_data[traffic_data[byte_col] > 0][byte_col]
    data_for_hist = data_for_hist[data_for_hist < data_for_hist.quantile(0.99)]
    
    plt.hist(data_for_hist, bins=50, edgecolor='black', color='lightgreen')
    threshold = data_for_hist.quantile(0.95)
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'95th Percentile ({threshold:.0f} bytes)')
    plt.xlabel('Transfer Size (bytes)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Data Transfer Size Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transfer_size_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir}/transfer_size_distribution.png")
else:
    print(f"  [!] Skipped - byte data not available")

# Visualization 3: Protocol Distribution
print("[*] Creating visualization 3/4: Protocol Distribution...")
plt.figure(figsize=(8, 6))
if 'proto' in traffic_data.columns:
    proto_counts = traffic_data['proto'].map(
        lambda x: protocol_map.get(x, f'Protocol {x}') if pd.notna(x) else 'Unknown'
    ).value_counts()
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    plt.pie(proto_counts.values[:5], labels=proto_counts.index[:5], autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('Network Protocol Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/protocol_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir}/protocol_distribution.png")
else:
    print(f"  [!] Skipped - protocol data not available")

# Visualization 4: ML Anomaly Scores
print("[*] Creating visualization 4/4: ML Anomaly Score Distribution...")
plt.figure(figsize=(10, 6))
normal = traffic_data[traffic_data['is_anomaly'] == 1]['anomaly_score']
anomaly = traffic_data[traffic_data['is_anomaly'] == -1]['anomaly_score']

plt.hist(normal, bins=50, alpha=0.7, label='Normal Traffic', color='blue')
plt.hist(anomaly, bins=50, alpha=0.7, label='Anomalous Traffic', color='red')
plt.xlabel('Anomaly Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Anomaly Score Distribution (Isolation Forest)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/anomaly_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {output_dir}/anomaly_scores.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("FINAL ANALYSIS SUMMARY".center(70))
print("=" * 70)

print(f"\n📊 Dataset Statistics:")
print(f"   Total Packets Analyzed: {len(traffic_data):,}")
if 'srcip' in traffic_data.columns:
    print(f"   Unique Source IPs: {traffic_data['srcip'].nunique():,}")
if 'dstip' in traffic_data.columns:
    print(f"   Unique Destination IPs: {traffic_data['dstip'].nunique():,}")

print(f"\n🔍 Detection Results:")
print(f"   Rule-Based Anomaly Categories: {anomalies_found}")
print(f"   ML-Detected Anomalies: {ml_anomalies:,} packets ({ml_anomalies/len(traffic_data)*100:.1f}%)")

if 'attack_cat' in traffic_data.columns or 'label' in traffic_data.columns:
    label_col = 'attack_cat' if 'attack_cat' in traffic_data.columns else 'label'
    if label_col == 'attack_cat':
        actual_attacks = (traffic_data[label_col] != 'Normal').sum()
    else:
        actual_attacks = (traffic_data[label_col] == 1).sum()
    
    print(f"   Actual Attacks in Dataset: {actual_attacks:,} ({actual_attacks/len(traffic_data)*100:.1f}%)")
    print(f"   Normal Traffic: {len(traffic_data) - actual_attacks:,} ({(len(traffic_data)-actual_attacks)/len(traffic_data)*100:.1f}%)")

if 'proto' in traffic_data.columns:
    print(f"\n📈 Protocol Breakdown:")
    proto_counts = traffic_data['proto'].value_counts()
    for proto_num, count in proto_counts.head(5).items():
        proto_name = protocol_map.get(proto_num, f'Protocol {proto_num}')
        print(f"   {proto_name}: {count:,} packets ({count/len(traffic_data)*100:.1f}%)")

print(f"\n📁 Output Files:")
print(f"   All visualizations saved in '{output_dir}/' folder")
print(f"   - attack_types_distribution.png")
print(f"   - transfer_size_distribution.png")
print(f"   - protocol_distribution.png")
print(f"   - anomaly_scores.png")

print("\n" + "=" * 70)
print("✅ Real-World Attack Analysis Complete!".center(70))
print("=" * 70)
print("\nNext Steps:")
print("1. Check the 'output_UNSW' folder for your visualization images")
print("2. Take screenshots of this terminal output")
print("3. Upload screenshots and images back to Claude")
print("4. Claude will create your final report with REAL data!")
print()
