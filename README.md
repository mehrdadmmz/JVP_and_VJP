# JVP_VJP 🚀🧮

A simple demonstration of **forward-mode** (JVP) and **reverse-mode** (VJP) automatic differentiation using [JAX](https://github.com/google/jax).

## Overview💫

- **Forward-Mode (JVP)**: Computes the directional derivative \(\frac{\partial f}{\partial x} \cdot v\) without explicitly forming the entire Jacobian.
- **Reverse-Mode (VJP)**: (Not fully shown in the example code yet) is typically used for gradient-based optimization and backpropagation.

In this repo, we:

1. Define a function \( f: \mathbb{R}^4 \to \mathbb{R}^3 \).
2. Use `jax.jacfwd(f)` to compute the full Jacobian at a specific point.
3. Demonstrate the **Jacobian-vector product (JVP)** using:
   - A naive approach: `full_jacobian @ multiplication_point`.
   - JAX’s built-in `jax.jvp`.

## Code Structure💻

- **`f(x)`**  
  A function mapping a 4D input to a 3D output, using element-wise exponentiation.

- **`main()`**  
  - Evaluates the Jacobian at a given point.  
  - Demonstrates the JVP both via manual matrix multiplication and via `jax.jvp`.

## Getting Started✅

1. **Clone the repo**:🚀
   ```bash
   git clone https://github.com/yourusername/JVP_VJP.git

2. **Install dependencies (make sure you have Python 3.7+):🚀
   ```bash
   import sys
   !{sys.executable} -m pip install jax jaxlib 

3. **Run the example:🚀
   ```bash 
   python main.py


