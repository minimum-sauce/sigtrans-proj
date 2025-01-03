#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import random
import string
import numpy as np
import matplotlib.pyplot as plt

def evaluate_transmissions(word: str, n: int = 10):
    """
    Calls simulation.py n times with the given 'word'.
    
    For each run, we record:
      - packet_correct (1 if entire word matches, else 0)
      - character error rate (CER) = char_errors / total_chars
    
    Returns:
      (packet_accuracy_mean, packet_accuracy_std, cer_mean, cer_std)

    Where:
      packet_accuracy_mean = mean of packet_correct * 100 (as %)
      packet_accuracy_std  = std of packet_correct * 100
      cer_mean            = mean of CER
      cer_std             = std of CER
    """
    script_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    packet_results = []  # Will store 1 if entire packet was correct, else 0
    cer_results = []     # Will store CER for each run

    for _ in range(n):
        # On Windows, "python" might be correct; on Unix, "python3" might be used.
        result = subprocess.run(["python", script_path, word],
                                capture_output=True, text=True)

        run_char_errors = 0
        run_total_chars = 0
        received = None

        for line in result.stdout.splitlines():
            if line.startswith("Received: "):
                received = line.replace("Received: ", "").strip()

            # e.g. "Char errors: 3, total chars: 6, CER: 0.500000"
            if line.startswith("Char errors: "):
                parts = line.split(',')
                try:
                    err_str  = parts[0].split(':')[1].strip()  # "3"
                    chars_str = parts[1].split(':')[1].strip() # "6"
                    run_char_errors = int(err_str)
                    run_total_chars = int(chars_str)
                except (IndexError, ValueError):
                    pass

        # Check if entire packet was correct
        if received == word:
            packet_results.append(1)
        else:
            packet_results.append(0)

        # Compute CER for this run
        if run_total_chars > 0:
            cer = run_char_errors / run_total_chars
        else:
            cer = 0.0
        cer_results.append(cer)

    # Convert packet correctness into accuracy (%) and compute mean/std
    packet_correct_array = np.array(packet_results, dtype=float)
    packet_accuracy_mean = packet_correct_array.mean() * 100.0
    packet_accuracy_std  = packet_correct_array.std(ddof=1) * 100.0  # ddof=1 => sample std

    # Compute mean and std for CER
    cer_array = np.array(cer_results, dtype=float)
    cer_mean = cer_array.mean()
    cer_std  = cer_array.std(ddof=1)  # sample std

    return (packet_accuracy_mean, packet_accuracy_std, cer_mean, cer_std)

def main():
    # amount_of_words_to_test: Amount of word lengths to test
    amount_of_words_to_test = 20
    lengths = list(range(1, amount_of_words_to_test + 1))
    # n: Number of times to run the simulation for each word length
    n = 50

    accuracies_mean = []
    accuracies_std = []
    cers_mean = []
    cers_std = []

    # Make random words reproducible if desired
    # random.seed(42)

    for length in lengths:
        # Generate a random word of 'length'
        test_word = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
        
        (acc_mean, acc_std, cer_mean, cer_std) = evaluate_transmissions(test_word, n=n)
        
        accuracies_mean.append(acc_mean)
        accuracies_std.append(acc_std)
        cers_mean.append(cer_mean)
        cers_std.append(cer_std)

        print(f"Word: {test_word} (length={length})")
        print(f"  Packet Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}")
        print(f"  CER            : {cer_mean:.6f} ± {cer_std:.6f}\n")

    plt.figure(figsize=(9, 4))
    # Subplot 1: Packet Accuracy
    plt.subplot(1, 2, 1)
    plt.errorbar(lengths, accuracies_mean, yerr=accuracies_std, fmt='o-', capsize=5)
    plt.ylim(0, 100)
    plt.xticks(lengths)
    plt.xlabel("Word Length")
    plt.ylabel("Accuracy (%)")
    plt.title("Packet Accuracy vs. Word Length")
    plt.grid(True)

    # Subplot 2: CER
    plt.subplot(1, 2, 2)
    plt.errorbar(lengths, cers_mean, yerr=cers_std, fmt='o-', capsize=5, color='red')
    plt.xticks(lengths)
    plt.xlabel("Word Length")
    plt.ylabel("CER")
    plt.title("Character Error Rate vs. Word Length")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig("my_plot.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
