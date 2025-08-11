# Natural Language Processing Summer 2025 Assignment | UniPi

![Python](https://img.shields.io/badge/python-3.12%2B-blue)

**Authors:** Giannakopoulos Nikolaos Ioannis

---

## Overview

This repository contains the coding portion of the 2025 Natural Language Processing (NLP) assignment at the University of Piraeus.

---

## Repository Structure
```
├── Paradoteo_1/     Python files covering the requirements of the first part
├── Paradoteo_2/     Python file covering the requirements of the second part
├── Documentation/   The documentation for the two exercises
├── myproject.toml   Poetry configuration
└── poetry.lock      Locked dependencies for reproducibility
```

---

## Prerequisites

- **Python** 3.12
- **Poetry/Conda** for dependency management

---

## Installation
1. **Create a new conda environment on `python12` using
```bash
conda create -n <env-name> python=3.12 -y
```

2. **Clone the repository**
   ```bash
   git clone https://github.com/nikolas-giannakopoulos/Natural-Language-Processing.git
   cd NLP2025
3. **Install dependencies**
  inside the `conda` environment
  ```bash
   pip install poetry
   poetry install --no-root
```
3. **Run the python file of your preference**