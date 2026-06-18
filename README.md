# 🚢 Titanic Survival Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0.3-yellow.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive machine learning solution for the Titanic - Machine Learning from Disaster competition on Kaggle.

## 📋 Table of Contents
- [🚢 Titanic Survival Prediction](#-titanic-survival-prediction)
  - [📋 Table of Contents](#-table-of-contents)
  - [📖 Project Overview](#-project-overview)
    - [🎯 Objective](#-objective)
    - [🔍 Key Insights](#-key-insights)
  - [📊 Dataset](#-dataset)
    - [Features Description](#features-description)
    - [Data Source](#data-source)
  - [🛠️ Technologies Used](#️-technologies-used)
    - [Core Libraries](#core-libraries)
    - [Development Tools](#development-tools)
  - [🚀 Getting Started](#-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
      - [1. Clone the Repository](#1-clone-the-repository)

## 📖 Project Overview

This project builds a predictive model to determine which passengers survived the Titanic shipwreck based on passenger characteristics. The Titanic sank on April 15, 1912, during her maiden voyage, resulting in the loss of 1,502 out of 2,224 passengers and crew.

### 🎯 Objective
Predict passenger survival (**0 = No, 1 = Yes**) using machine learning algorithms.

### 🔍 Key Insights
- Women had significantly higher survival rates
- 1st class passengers had priority in lifeboats
- Children and elderly were given preference
- Family size played a crucial role in survival

## 📊 Dataset

### Features Description

| Feature | Description | Values |
|---------|-------------|--------|
| **survival** | Survival status (Target) | 0 = No, 1 = Yes |
| **pclass** | Ticket class | 1st, 2nd, 3rd |
| **sex** | Gender | male, female |
| **age** | Age in years | Float value |
| **sibsp** | # of siblings/spouses aboard | Integer |
| **parch** | # of parents/children aboard | Integer |
| **ticket** | Ticket number | String |
| **fare** | Passenger fare | Float value |
| **cabin** | Cabin number | String |
| **embarked** | Port of embarkation | C, Q, S |

### Data Source
- **Training Data**: `train.csv` (891 passengers with survival labels)
- **Test Data**: `test.csv` (418 passengers without survival labels)
- **Sample Submission**: `gender_submission.csv`

[Download Dataset from Kaggle](https://www.kaggle.com/competitions/titanic/data)

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization
- **Scikit-learn** - Machine learning algorithms

### Development Tools
- **Jupyter Notebook** - Interactive development
- **Git** - Version control
- **pip** - Package management

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional)

### Installation Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/titanic-project.git
cd titanic-project