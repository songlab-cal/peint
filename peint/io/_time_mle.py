import os
from typing import List
import pickle

def read_output_stats(input_path: str):
    # outputs[i][0] gives the tuple of x, y, t_mle, t_wag, nll_f, nll_b for transition i
    outputs = []
    with open(input_path, "rb") as infile:
        while True:
            try:
                outputs.extend(pickle.load(infile))
            except EOFError:
                break
    return outputs

def write_output_stats(output: List, output_path: str):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "wb") as outfile:
        pickle.dump(output, outfile)

def write_new_transitions(transitions: List, output_path: str):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    res = (
        f"{len(transitions)} transitions\n"
        + "\n".join([f"{x} {y} {t_mle}" for (x, y, t_mle) in transitions])
        + "\n"
    )
    try:
        with open(output_path, "w") as outfile:
            outfile.write(res)
            outfile.flush()
    except Exception as e:
        print(f"Failed to write new transitions to {output_path} due to {e}")

def write_nlls(nlls: List[float], output_path: str):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, 'w') as f:
        f.write(f"{len(nlls)} transitions\n")
        for nll in nlls:
            f.write(f"{nll}\n")

def read_nlls(output_path: str):
    nlls = []
    with open(output_path, 'r') as f:
        f.readline()
        nlls.extend(float(line.strip()) for line in f) 
    return nlls 