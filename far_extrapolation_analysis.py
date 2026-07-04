# -*- coding: utf-8 -*-
"""FAR Extrapolation Analysis (Defense 1 & 2)

This script calculates and visualizes the False Attribution Rate (FAR) 
as the codebook size (N) scales up to planetary levels (e.g., millions).
It uses the Binomial CDF (Hamming sphere approximation) to show that 
Nearest-Neighbor matching (tolerating bit errors) inevitably leads to 
100% FAR collisions at scale, whereas Exact-Match remains highly secure.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def plot_far_extrapolation(payload_bits=64, save_path="results/far_extrapolation.png"):
    """
    Plots FAR vs Codebook Size (N) for different tolerated Hamming distances.
    We use 64 bits here to represent a planetary-scale payload capacity.
    """
    # N candidates: 10^2 to 10^8
    N_values = np.logspace(2, 8, num=100)
    
    # Tolerated distances (simulating Nearest-Neighbor with typical BERs)
    # 0% BER (Exact Match), 5% BER (d=3), 10% BER (d=6), 25% BER (d=16)
    tolerances = [0, 3, 6, 16]
    labels = ["Exact Match (0% BER)", "NN (5% BER Tol.)", "NN (10% BER Tol.)", "NN (25% BER Tol.)"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    
    plt.figure(figsize=(10, 6))
    
    for d, label, color in zip(tolerances, labels, colors):
        # Probability that a random candidate falls within Hamming distance d
        p_collision = binom.cdf(d, payload_bits, 0.5)
        
        # FAR = Probability of AT LEAST ONE false match among N candidates
        far = 1.0 - (1.0 - p_collision)**N_values
        
        plt.plot(N_values, far, label=f"{label} (d={d})", color=color, linewidth=2.5)
    
    plt.xscale("log")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Codebook Size (N) - Log Scale", fontsize=12)
    plt.ylabel("False Attribution Rate (FAR)", fontsize=12)
    plt.title(f"Planetary-Scale FAR Extrapolation ({payload_bits}-bit Payload)", fontsize=14, fontweight="bold")
    plt.axvline(x=1e6, color="gray", linestyle="--", alpha=0.7)
    plt.text(1.2e6, 0.5, "N = 1 Million\n(Planetary Scale)", color="gray")
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=11)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"FAR Extrapolation Plot saved to {save_path}")

if __name__ == "__main__":
    # 1. 64-bit payload extrapolation (to justify planetary scale)
    plot_far_extrapolation(payload_bits=64, save_path="results/far_extrapolation_64bit.png")
    
    # 2. 16-bit payload extrapolation (our current model constraint)
    # Note: 16-bit capacity maxes out at 65,536, so anything beyond N=10^4 rapidly collides.
    # It demonstrates why ECC-8bit is even worse (capacity 256).
    plot_far_extrapolation(payload_bits=16, save_path="results/far_extrapolation_16bit.png")
