# Zero-Knowledge Proof (ZKP) Learning Path

This repository is a curated learning path for understanding Zero-Knowledge Proofs (ZKPs). It provides a structured collection of resources, tutorials, and implementation in Python to guide learners from the fundamentals of zero-knowledge concepts to advanced applications, including cryptographic implementations, zk-SNARKs, and zk-STARKs.

## Getting Started

### Prerequisites

This project requires Python 3.7+ and Jupyter Notebook/Lab.

### Installation

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
pip install numpy sympy galois matplotlib jupyter
```

Or if a `requirements.txt` file is available:
```bash
pip install -r requirements.txt
```

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

**Dependencies:**
- numpy (v1.21.5 or compatible)
- sympy (v1.10.1 or compatible)
- galois
- matplotlib (v3.5.1 or compatible)
- merkle (custom module included in this repository)

#### circleSTARK.ipynb

This notebook implements Circle STARK, a variant of the standard STARK proof system that uses the circle group over a prime field instead of a multiplicative subgroup. It demonstrates:
- Circle group arithmetic over finite fields
- Using Mersenne primes (like M₃₁) for efficient FFT domains
- Circle STARK protocol implementation

To run:
```bash
jupyter notebook circleSTARK.ipynb
```

**Dependencies:**
- numpy
- matplotlib
- merkle (custom module included in this repository)

### Notes

- Both notebooks use the `merkle.py` module for Merkle tree implementation (based on StarkWare's implementation)
- Make sure all dependencies are installed before running the notebooks
- The notebooks include detailed explanations and can be run cell-by-cell for better understanding
