import argparse
import json
import os
import torch
import pickle
from Bio import SeqIO
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import logging
import esm
from tqdm import tqdm

from typing import List
from protevo.models._flash_esm import ESM2Flash
from protevo.utils import amino_acids

from protevo.models._transformer import (
    ProtEvoPretrainedTransformerModule,
    PeintEvaluator
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Distributed sequence comparison using GPU')
    parser.add_argument('--dist-file', type=str, required=True,
                       help='Path to the GPU distribution JSON file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing sequence files')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for model inference')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to save results')
    parser.add_argument('--gpu-id', type=int, required=True,
                       help='GPU ID to use for this process')
    parser.add_argument('--time-dict', type=str, required=False,
                       help='Path to JSON file containing pairwise time values')
    parser.add_argument('--default-time', type=float, default=1.0,
                       help='Default time value if no time dictionary is provided or pair not found')
    return parser.parse_args()

def load_model(model_checkpoint_path: str, device: torch.device):
    """
    Load the model from checkpoint.
    Replace this with your actual model loading logic.
    """

    vocab = esm.data.Alphabet.from_architecture("ESM-1b")

    esm_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    flash_esm = ESM2Flash(
        num_layers = esm_model.num_layers,
        embed_dim = esm_model.embed_dim,
        attention_heads = esm_model.attention_heads,
        alphabet ='ESM-1b',
        token_dropout = True,
        dropout_p = 0.0
    )

    flash_esm.load_state_dict(esm_model.state_dict(), strict=False)
    del(esm_model)

    sd = torch.load(model_checkpoint_path, map_location='cpu')
    module = ProtEvoPretrainedTransformerModule(
        esm_model = flash_esm,
        esm_vocab = vocab,
        **sd['hyper_parameters']
    )

    module.load_state_dict(sd['state_dict'])

    model = PeintEvaluator(
        esm_model = flash_esm,
        esm_vocab = vocab,
        **sd['hyper_parameters']
    )

    model.load_state_dict(module.model.state_dict())

    return model.eval().to(device), vocab

def load_all_sequences(data_dir: str, required_files: set) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load all required sequence files at once
    Returns a dictionary mapping filename to list of (description, sequence) pairs
    """
    sequences = {}
    logging.info(f"Loading sequences from {len(required_files)} files...")
    for filename in tqdm(required_files):
        infile = filename + '_conforming.fasta'
        file_path = os.path.join(data_dir, infile)
        sequences[filename] = [(r.description, str(r.seq)) for r in SeqIO.parse(file_path, "fasta")]
    return sequences

def load_time_dictionary(time_dict_path: str) -> Dict[Tuple[str, str], float]:
    """
    Load time dictionary from JSON file and format it for easy lookup
    """
    with open(time_dict_path, 'r') as f:
        raw_dict = json.load(f)
    
    # Convert to tuple keys for easier lookup
    formatted_dict = {}
    for file1, inner_dict in raw_dict.items():
        for file2, time_value in inner_dict.items():
            formatted_dict[(file1, file2)] = float(time_value)
            # Also store reverse lookup if not already present
            if (file2, file1) not in formatted_dict:
                formatted_dict[(file2, file1)] = float(time_value)
    
    return formatted_dict

def group_comparisons(comparisons: List[Tuple[Tuple[str, int], Tuple[str, int]]]) -> Dict[Tuple[str, int], List[Tuple[str, int]]]:
    """
    Group comparisons by reference sequence to minimize forward passes
    Returns a dictionary mapping reference sequence to list of sequences to compare against
    """
    grouped = defaultdict(list)
    for (ref_file, ref_idx), (test_file, test_idx) in comparisons:
        grouped[(ref_file, ref_idx)].append((test_file, test_idx))
    return grouped

def get_time_value(ref_file: str, test_file: str, time_dict: Dict[Tuple[str, str], float], default_time: float) -> float:
    """
    Get time value for a pair of files from the time dictionary
    """
    return time_dict.get((ref_file, test_file), default_time)

def process_comparisons(model, reference_seq: str, test_seqs: List[str], 
                       times: List[float], device: torch.device, batch_size: int) -> torch.Tensor:
    """
    Process all comparisons for a single reference sequence at once
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            likelihoods = model.evaluate_likelihood(
                x=reference_seq,
                y=test_seqs,
                t=times,
                device=device,
                batch_size=batch_size
            )
    return likelihoods

def main():
    args = parse_args()
    device = torch.device('cuda')
    
    # Load time dictionary if provided
    time_dict = {}
    if args.time_dict:
        logging.info(f"Loading time dictionary from {args.time_dict}")
        time_dict = load_time_dictionary(args.time_dict)
    
    # Load GPU distribution file
    with open(args.dist_file, 'r') as f:
        gpu_comparisons = json.load(f)[args.gpu_id]
    
    # Get all unique files needed
    required_files = set()
    for (ref_file, _), (test_file, _) in gpu_comparisons:
        required_files.add(ref_file)
        required_files.add(test_file)
    
    # Load all sequences at once
    all_sequences = load_all_sequences(args.data_dir, required_files)
    
    # Load model
    logging.info(f"Loading model from {args.checkpoint}")
    model, vocab = load_model(args.checkpoint, device)
    
    # Group comparisons by reference sequence
    grouped_comparisons = group_comparisons(gpu_comparisons)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Process each reference sequence and all its comparisons at once
    for (ref_file, ref_idx), test_sequences in tqdm(grouped_comparisons.items(), 
                                                   desc=f"GPU {args.gpu_id} Processing"):
        # Get reference sequence
        reference_seq = all_sequences[ref_file][ref_idx][1]
        ref_description = all_sequences[ref_file][ref_idx][0]
        
        # Prepare all test sequences and their times for this reference
        test_seqs = []
        test_times = []
        test_info = []  # Store file and index info for results
        
        for test_file, test_idx in test_sequences:
            test_seqs.append(all_sequences[test_file][test_idx][1])
            test_times.append(get_time_value(ref_file, test_file, time_dict, args.default_time))
            test_info.append((test_file, test_idx, all_sequences[test_file][test_idx][0]))
        
        # Process all comparisons for this reference sequence at once
        likelihoods = process_comparisons(
            model=model,
            reference_seq=reference_seq,
            test_seqs=test_seqs,
            times=test_times,
            device=device,
            batch_size=args.batch_size
        )
        
        # Store results
        for i, (test_file, test_idx, test_description) in enumerate(test_info):
            key = f"{ref_file}_{ref_idx}_{test_file}_{test_idx}"
            results[key] = {
                'reference': ref_description,
                'test': test_description,
                'likelihood': likelihoods[i],
                'time': test_times[i]
            }
    
    # Save results
    output_file = output_dir / f"results_gpu_{args.gpu_id}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()