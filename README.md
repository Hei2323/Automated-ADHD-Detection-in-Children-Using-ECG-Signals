# ADHD_detection：Automated ADHD Detection in Children Using ECG Signals

**Overview**

This project proposes an automated artificial intelligence (AI) pipeline for detecting Attention Deficit Hyperactivity Disorder (ADHD) in children based on single-lead electrocardiogram (ECG) signals. ADHD, a prevalent neurodevelopmental disorder, is often underdiagnosed or misdiagnosed due to subjective behavioral assessments and limited access to specialists. Our AI pipeline aims to address these challenges by leveraging cost-effective, widely accessible ECG data for accurate ADHD detection.

**Key Features**
**AI-Powered Detection:** Utilizes a one-dimensional convolutional neural network (1D-CNN) for deep feature extraction and machine learning classifiers like Random Forest (RF) for accurate ADHD classification.

**Real-World Validation:** The model was tested on large-scale, age-matched datasets (8,644 cases) and a real-world cohort (99,980 cases) with a 10% ADHD prevalence, achieving an impressive 99.42% accuracy and 96.54% recall rate.

**Objective and Automated:** The system uses raw ECG signals with no manual feature extraction, offering an unbiased, scalable solution to ADHD screening.

**Methodology**

The pipeline includes three primary steps:

**1D-CNN for Feature Extraction:** The raw ECG signals are processed by a deep neural network to identify patterns associated with ADHD.

**Score-CAM for Interpretability and Feature Extraction:** We use Score-Weighted Class Activation Mapping (Score-CAM) to visually highlight ECG segments critical for ADHD detection, enhancing model transparency, and extract features from the activate map.

**Machine Learning Classifiers:** The extracted features are fed into classifiers like Random Forest, SVM, and Logistic Regression to predict ADHD status.

**Model Performance**

Healthy-Control Cohort (8,644 cases): The model achieved 95.20% accuracy and 95.15% recall.

Real-World Cohort (99,980 cases): The re-trained pipeline reached 99.42% accuracy and 96.54% recall, demonstrating robust performance in real-world settings.

**Potential Impact**

Improved Diagnostic Access: This AI-powered system could be deployed in remote or low-resource areas, reducing dependency on specialized psychiatric evaluations and increasing early ADHD detection.

Gender and Socioeconomic Equity: The automated approach helps address diagnostic biases that disproportionately affect girls and socioeconomically disadvantaged children.

**Future Directions**

Multicenter Validation: Further testing across diverse populations will be crucial to validate the model’s generalizability.

Integration with Wearables: Future iterations may leverage ECG data from wearable devices, facilitating real-time ADHD monitoring and early intervention.

**Conclusion**

This project presents a promising, low-cost, and automated method for ADHD detection using single-lead ECG signals. The proposed AI pipeline could transform the efficiency and accessibility of ADHD diagnosis globally, providing an essential tool for pediatric healthcare providers.
