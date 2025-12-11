# Network Anomaly Detection Using UNSW-NB15 Dataset

A comprehensive network security system implementing hybrid anomaly detection using rule-based heuristics and machine learning (Isolation Forest) on the UNSW-NB15 dataset.

## Features

- Dual detection methodology: Rule-based + Machine Learning
- Analysis of 50,000+ network packets from UNSW-NB15 dataset
- Detection of 9 attack categories (Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms)
- Real-time visualization generation
- Performance metrics: 27.6% precision, 1,378 true positives

## Requirements
```
pandas
numpy
matplotlib
scikit-learn
```

## Installation
```bash
# Install required libraries
pip install pandas numpy matplotlib scikit-learn --break-system-packages

# Or using apt (Kali Linux)
sudo apt install python3-pandas python3-numpy python3-matplotlib python3-sklearn
```

## Dataset

Download UNSW-NB15 dataset from Kaggle:
- Dataset: [UNSW-NB15 on Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
- Required file: `UNSW_NB15_training-set.csv`
- Place in the same directory as the script

## Usage
```bash
python3 anomaly_detector_UNSW.py
```

## Output

The script generates:
- 4 visualization charts in `output_UNSW/` folder
- Terminal output with detection statistics
- Protocol distribution analysis
- Attack type categorization

## Results

- **Total Packets Analyzed:** 50,000
- **Attacks Detected:** 27,448 (54.9%)
- **Normal Traffic:** 22,552 (45.1%)
- **ML Detection Rate:** 10.0% (5,000 anomalies)
- **True Positives:** 1,378
- **Detection Accuracy:** 27.6%

## Attack Categories Detected

1. Generic - 11,369 packets
2. Exploits - 6,801 packets
3. Fuzzers - 3,685 packets
4. DoS - 2,413 packets
5. Reconnaissance - 2,080 packets
6. Analysis - 398 packets
7. Backdoor - 378 packets
8. Shellcode - 243 packets
9. Worms - 33 packets

## Visualizations

The system generates 4 professional charts:
- Attack Types Distribution
- Data Transfer Size Distribution
- Network Protocol Distribution
- Anomaly Score Distribution

## License

Educational project for network security coursework.

## Author

Amila Niroshana Thilakarathne - Network Security Implementation
