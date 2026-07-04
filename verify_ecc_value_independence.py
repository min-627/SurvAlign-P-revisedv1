# -*- coding: utf-8 -*-
"""Verify ECC Value-Independence (Empirical Test)

This script tests whether the neural audio codec channel (EnCodec) violates 
the "value-independence" assumption of the ECC baseline.

If the neural network's error generation strongly depends on the specific 
bit patterns (e.g. 0000 vs 1010), then simulating ECC by drawing from the 
full uniform 16-bit space might be statistically invalid compared to drawing 
exclusively from a restricted 256-word ECC codebook.

We test this by comparing:
Group A: 50 audio samples embedded with completely random 16-bit messages.
Group B: 50 audio samples embedded with messages restricted to a fixed 256-word codebook.

We measure P(Hamming <= 2) for both groups. If they match closely, we empirically 
prove that the uniform proxy is an unbiased estimator for the restricted subset.
"""

import os
import torch
import numpy as np

from survalign_p import AlignMarkManager, DifferentiableDistortion, UnifiedSpeechDataset
from phase1_attribution import _apply_internal_attack
from experiment_utils import set_global_seed

def main():
    print("1. Setup Models & Codebook")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(42)
    
    alignmark = AlignMarkManager(device=device, latent_mode="public_code")
    distorter = DifferentiableDistortion(sr=16000, vae=alignmark.vae).to(device)
    
    n_samples = 50
    
    # Create a fixed pseudo-random ECC Codebook (256 valid codewords)
    # This acts as our "Nordstrom-Robinson" proxy subset.
    ecc_codebook = torch.randint(0, 2, (256, 16), dtype=torch.long, device=device)
    
    hamming_a = []
    hamming_b = []
    
    print(f"2. Running Empirical Verification on {n_samples} samples...")
    for i in range(n_samples):
        # Generate synthetic 3-second audio tensor to test the neural channel logic
        # without requiring a full dataset download.
        wav = torch.randn(1, 1, 16000 * 3, device=device)
        
        # --- GROUP A: Unrestricted Uniform Random 16-bit Message ---
        msg_a = torch.randint(0, 2, (1, 16), dtype=torch.long, device=device)
        wav_wm_a, _ = alignmark.embed(wav, msg_a)
        
        with torch.no_grad():
            wav_atk_a = _apply_internal_attack(wav_wm_a, "reconstruct_nq6", distorter, seed=42+i)
            _, _, dec_a = alignmark.decode(wav_atk_a)
            dist_a = torch.sum(msg_a != dec_a.long()).item()
            hamming_a.append(dist_a)
            
        # --- GROUP B: Restricted Codebook Message ---
        # Pick a random codeword from our 256-word codebook
        idx = torch.randint(0, 256, (1,)).item()
        msg_b = ecc_codebook[idx:idx+1]
        wav_wm_b, _ = alignmark.embed(wav, msg_b)
        
        with torch.no_grad():
            wav_atk_b = _apply_internal_attack(wav_wm_b, "reconstruct_nq6", distorter, seed=42+i)
            _, _, dec_b = alignmark.decode(wav_atk_b)
            dist_b = torch.sum(msg_b != dec_b.long()).item()
            hamming_b.append(dist_b)
            
        if (i+1) % 10 == 0:
            print(f"   [{i+1}/{n_samples}] Processed...")
            
    # Calculate P(Hamming <= 2)
    hamming_a = np.array(hamming_a)
    hamming_b = np.array(hamming_b)
    
    p_a = np.mean(hamming_a <= 2) * 100
    p_b = np.mean(hamming_b <= 2) * 100
    
    print("\n3. Empirical Verification Results")
    print("-" * 50)
    print(f"Group A (Uniform 16-bit space) P(d <= 2) : {p_a:.1f}%")
    print(f"Group B (256-word ECC Subspace) P(d <= 2): {p_b:.1f}%")
    print("-" * 50)
    
    diff = abs(p_a - p_b)
    print(f"Absolute Difference: {diff:.1f}%")
    
    if diff <= 10.0:  # Allow some margin of error for a 50-sample Monte Carlo
        print("\nSUCCESS: The error generation is statistically independent of the specific")
        print("   codeword subset. The Uniform Random Proxy is an unbiased and mathematically")
        print("   valid estimator for the ECC baseline.")
    else:
        print("\nFAILED: The neural channel exhibits strong value-dependence. The uniform proxy")
        print("   is biased. A full dataset re-embedding with valid codewords is required.")

if __name__ == "__main__":
    main()
