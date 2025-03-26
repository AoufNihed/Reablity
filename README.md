# Reablity: Reliability Modeling & AI-driven Analysis

## Overview

Reablity is a comprehensive system for analyzing and improving the reliability of electrical networks. It integrates classical reliability models such as Fault Tree Analysis (FTA), Mean Time to Repair (MTTR), and Markov Chains with machine learning techniques to assess the quality, stability, and continuity of electrical grids.

The objectives of this project include:
- Modeling classical reliability systems using probabilistic methods.
- Enhancing predictions with machine learning techniques such as Random Forest, K-Means, and Isolation Forest.
- Developing an intelligent multi-agent system using CrewAI for continuous monitoring and optimization of the power supply.
- Providing data visualizations for failure detection and prevention.

## Project Structure

The project is divided into three main components:

- `classic/` : Implements traditional reliability algorithms, including MTTR, FTA, and Markov Chains.
- `ml/` : Contains machine learning models for fault detection and classification.
- `crew_ai/` : Develops a multi-agent system for real-time stability, quality assessment, and anomaly detection.

## Installation Guide

### Prerequisites

Ensure that Python 3.8 or later is installed:
```bash
python --version
```

Ensure that Git is installed:
```bash
git --version
```

### Cloning the Repository

```bash
git clone https://github.com/AoufNihed/Reablity.git
cd Reablity
```

### Installing Dependencies

Install the required Python libraries:
```bash
pip install pandas matplotlib numpy scikit-learn crewai
```

For better package management, create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Modules

### Classical Reliability Models

Navigate to the `classic/` folder and execute the algorithms:
```bash
cd classic
python mttr.py  # Mean Time to Repair model
python fta.py   # Fault Tree Analysis
python markov.py # Markov Chains analysis
```

### Machine Learning-Based Fault Detection

Move to the `ml/` folder and run the models:
```bash
cd ../ml
python xgboost_model.py
python random_forest.py
python kmeans_clustering.py
```

### Multi-Agent System (MAS) with CrewAI

Navigate to the `crew_ai/` folder and execute the main script:
```bash
cd ../crew_ai
python main.py
```

## Future Enhancements

- Integration with real-world electrical grid data.
- Optimization of fault detection models for higher accuracy.
- Enhancement of multi-agent communication using CrewAI.
- Deployment on cloud or edge devices for real-time monitoring.

## Project Details

This project is a mini-project for the **Power Electrical Engineering** program at **ESGEE**, under the **ARTD module**. Developed and coded by **Nihed Aouf**.
