# Zero-Knowledge Proof (ZKP) Learning Path

This repository is a curated learning path for understanding Zero-Knowledge Proofs (ZKPs). It provides a structured collection of resources, tutorials, and implementation in Python to guide learners from the fundamentals of zero-knowledge concepts to advanced applications, including cryptographic implementations, zk-SNARKs, and zk-STARKs.

## Getting Started

### Prerequisites

This project requires Python 3.7+ and Jupyter Notebook/Lab.

### Installation

#### Option 1: Using Dev Container (Recommended)

The easiest way to get started with a consistent development environment and **automatic port forwarding**.

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running
- VS Code or Cursor IDE with the ["Dev Containers"](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

**Steps:**

1. Clone the repository:
```bash
git clone <repository-url>
cd zeroknowledge
```

2. Open the project in VS Code/Cursor:
```bash
code .  # For VS Code
# or
cursor . # For Cursor IDE
```

3. When prompted, click **"Reopen in Container"**, or manually:
   - Press `Cmd/Ctrl + Shift + P`
   - Select **"Dev Containers: Reopen in Container"**

4. Wait for the container to build (first time only)

#### Option 2: Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd zeroknowledge
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Locked Dependencies (see `requirements.txt`):**
- numpy==2.4.6
- sympy==1.14.0
- galois==0.4.11
- matplotlib==3.10.9
- jupyter==1.1.1
- notebook==7.3.3
- merkle (custom module included in this repository as `merkle.py`)

### Running the Notebooks

#### zkSTARK.ipynb

This notebook demonstrates the implementation of a zk-STARK (Zero-Knowledge Scalable Transparent Argument of Knowledge) proof system. It includes step-by-step examples of:
- Finite field arithmetic
- Polynomial constraints
- FRI (Fast Reed-Solomon Interactive Oracle Proof of Proximity) protocol
- Merkle tree commitments

To run:
```bash
jupyter notebook zkSTARK.ipynb
```

#### circleSTARK.ipynb

This notebook implements Circle STARK, a variant of the standard STARK proof system that uses the circle group over a prime field instead of a multiplicative subgroup. It demonstrates:
- Circle group arithmetic over finite fields
- Using Mersenne primes (like M₃₁) for efficient FFT domains
- Circle STARK protocol implementation

To run:
```bash
jupyter notebook circleSTARK.ipynb
```

### Running Jupyter in Dev Container

If using the Dev Container, Jupyter is already configured. You can:

1. **Open notebooks directly in VS Code/Cursor**: 
   - Simply click on any `.ipynb` file, and it will open with Jupyter support

2. **Start Jupyter Notebook server**:
   ```bash
   jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
   ```
   The port 8888 is automatically forwarded, so you can access it at `http://localhost:8888`
