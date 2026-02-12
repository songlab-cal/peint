from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import multiprocessing
import tqdm
import os
import esm

from protevo.datasets import PeintDataset, PeintCollator
from protevo import caching as protevo_caching
from protevo.caching import secure_parallel_output
from protevo.utils import amino_acids, get_process_args
from protevo import io
from protevo.models import PeintTransformer
from protevo.models.training import PeintLightningModule
from protevo.models._flash_esm import ESM2Flash
from protevo.models._loading import load_model

def load_peint_esm2_150M(
    model_path,
    embed_dim=640,
    num_heads=20,
    num_encoder_layers=5,
    num_decoder_layers=5,
    device='cuda',
    eval=True
):
    default_esm, default_vocab = esm.pretrained.esm2_t30_150M_UR50D() #uses the 150M model
    esmmodel = ESM2Flash() #ESM2 rewritten using Flash Atten
    esmmodel.load_state_dict(default_esm.state_dict(), strict=False)

    model = PeintTransformer(
        esm_model=esmmodel,
        esm_vocab=default_vocab,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )

    sd = torch.load(model_path, map_location='cpu')

    module = PeintLightningModule(
        esm_model = esmmodel,
        esm_vocab = default_vocab,
        **sd['hyper_parameters']
    )

    # Load the checkpoint
    module.load_state_dict(sd['state_dict'])
    pretrained_model = module.model
    model.load_state_dict(pretrained_model.state_dict()) #transfer weights
    if eval:
        model = model.eval()
    model = model.to(device)

    return model, default_vocab

def _time_mle(
    model, 
    family, 
    vocab, 
    transitions_dir,
    output_transitions_dir,
    max_len: int = 1022, 
    device: str = 'cuda', 
    batch_size=32,
    num_steps=80,
    lr=1e-1,
    gamma=0.99,
    initializer=0.6,
):
    model = model.to(device)
    eps=5e-3
    output_transitions_path = os.path.join(output_transitions_dir, family + ".txt")
    ds = PeintDataset(
        data_path = transitions_dir,
        vocab = vocab,
        families = [family],
        max_len = max_len
    )

    collator = PeintCollator(
        vocab = vocab,
        mask_prob = 0,  # don't mask x
    )

    dataloader = DataLoader(ds, shuffle=False, collate_fn=collator, batch_size=batch_size)
    
    transitions = []
    
    for _, batch in enumerate(dataloader):
        x, y, x_t, y_t, t, xmask, ymask = batch
        # Each (x, y, t) transition is also a (y, x, t) transition, so this guarantees we see each (x, y, t) triplet exactly once
        batch_f = [tensor[::2, ...] for tensor in batch]  # 0th, 2nd, 4th, ...
        batch_r = [tensor[1::2, ...] for tensor in batch]  # 1st, 3rd, 5th, ...
        t_wag = t[::2, ...].detach().clone()

        # Initialize it agnostic of WAG t: this is useful for applications where we don't already have somewhere to seed an estimate
        t_guesses = torch.full_like(t_wag, initializer, requires_grad=True)
        optimizer = torch.optim.Adam([t_guesses], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        for _ in range(num_steps):
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                optimizer.zero_grad()

                #forward and reverse nll are unreduced nll's across the whole sequence
                x, x_t, y, y_t, t, xmask, ymask = batch_f
                forward = [b.to(device) for b in [x, x_t, y, y_t, t_guesses, xmask, ymask]]
                x, x_t, y, y_t, t, xmask, ymask = forward
                x_logits, y_logits = model(x, y, t, xmask, ymask)
                forward_nlls = F.cross_entropy(y_logits.transpose(-1,-2), y_t, reduction='none', ignore_index = vocab.padding_idx)
                mean_forward_time_nll = forward_nlls.sum(1) 

                x, x_t, y, y_t, t, xmask, ymask = batch_r 
                reverse = [b.to(device) for b in [x, x_t, y, y_t, t_guesses, xmask, ymask]]
                x, x_t, y, y_t, t, xmask, ymask = reverse
                x_logits, y_logits = model(x, y, t, xmask, ymask)
                reverse_nlls = F.cross_entropy(y_logits.transpose(-1,-2), y_t, reduction='none', ignore_index = vocab.padding_idx)
                mean_reverse_time_nll = reverse_nlls.sum(1)
                
                loss = torch.mean(mean_forward_time_nll + mean_reverse_time_nll) # mean joint nll across the whole sequence, evaluated at the given time
                loss.backward()

                optimizer.step()
                scheduler.step()
                t_guesses.data.clamp_(min=eps)
                
        t_mles = [t.item() for t in t_guesses]
        x_seqs = convert_batched_tokens_to_sequences(x, vocab, aa_only=True)
        y_seqs = convert_batched_tokens_to_sequences(y, vocab, aa_only=True)

        forward_batches = [(x_f, y_f, t_f) for (x_f, y_f, t_f) in zip(x_seqs, y_seqs, t_mles)]
        reverse_batches = [(y_r, x_r, t_r) for (y_r, x_r, t_r) in zip(y_seqs, x_seqs, t_mles)]

        for f_batch, r_batch in zip(forward_batches, reverse_batches):
            transitions.append(r_batch)
            transitions.append(f_batch)

    io.write_new_transitions(transitions, output_transitions_path)
    secure_parallel_output(output_transitions_dir, family)

def _map_func_time_mle(args: List) -> None:
    train_transitions_dir, test_transitions_dir, families, model, vocab, output_train_transitions_dir,  output_test_transitions_dir, max_len, batch_size, num_steps, lr, gamma, initializer = args

    data_paths = [train_transitions_dir, test_transitions_dir]
    output_dirs = [output_train_transitions_dir, output_test_transitions_dir]

    for family in families:
        for data_dir, output_transitions_dir in zip(data_paths, output_dirs):   
            _time_mle(
                model=model,
                family=family,
                transitions_dir=data_dir,
                output_transitions_dir=output_transitions_dir,
                vocab=vocab,
                max_len=max_len,
                batch_size=batch_size,
                num_steps=num_steps,
                lr=lr,
                gamma=gamma,
                initializer=initializer,
                output_train_transitions_dir = output_train_transitions_dir, 
                output_test_transitions_dir = output_test_transitions_dir
            )

@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args = ['num_processes', 'batch_size'],
    exclude_args_if_default = ["max_len", "num_steps", "lr" , "gamma"],
    output_dirs=["output_transitions_dir"],
    write_extra_log_files=True
)
def estimate_transition_times(
    transitions_dir: str,
    families: List[str],
    model_checkpoint_path: str,
    max_len: int = 1022,
    batch_size: int = 32,
    num_steps: int = 80,
    lr: float = 1e-1,
    gamma: float = 0.99,
    initializer: float = 0.6,
    num_processes: int = 1,
    output_transitions_dir: Optional[str] = None):

    device = torch.device('cuda')
    
    model, vocab = load_model(
        model_checkpoint_path = model_checkpoint_path,
        use_cached_model = False,
        device = device
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Going to reestimate transition times for {len(families)} families")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == "cuda", "cuda is unavailable on this machine"

    for family in tqdm.tqdm(families):
        
        _time_mle(
            model=model,
            family=family,
            vocab =vocab,
            transitions_dir=transitions_dir,
            output_transitions_dir=output_transitions_dir,
            max_len=max_len,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr,
            gamma=gamma,
            initializer=initializer,
        )


@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    output_dirs=["output_train_transitions_dir", "output_test_transitions_dir"],
    write_extra_log_files=True
)
def reestimate_transition_times(
    train_transitions_dir: str,
    test_transitions_dir: str,
    families: List[str],
    num_processes: int,
    model_path: str,
    max_len=1022,
    batch_size=32,
    num_steps=80,
    lr=1e-1,
    gamma=0.99,
    initializer=0.6,
    embed_dim=640,
    num_heads=20,
    num_encoder_layers=5,
    num_decoder_layers=5,
    output_train_transitions_dir: Optional[str] = None, 
    output_test_transitions_dir: Optional[str] = None, 
):
    logger = logging.getLogger(__name__)
    logger.info(f"Going to reestimate transition times for {len(families)} families")

    if not os.path.exists(train_transitions_dir):
        raise ValueError(f"Could not find train_transitions_dir {train_transitions_dir}")

    if not os.path.exists(test_transitions_dir):
        raise ValueError(f"Could not find test_transitions_dir {test_transitions_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == "cuda", "cuda is unavailable on this machine"
        
    model, default_vocab = load_peint_esm2_150M(
        model_path=model_path,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        device=device
    )

    map_args = [
        [
            train_transitions_dir,
            test_transitions_dir,
            get_process_args(process_rank, num_processes, families),
            model,
            default_vocab,
            output_train_transitions_dir,
            output_test_transitions_dir,
            max_len,
            batch_size,
            num_steps,
            lr,
            gamma,
            initializer
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_map_func_time_mle, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_map_func_time_mle, map_args),
                total=len(map_args),
            )
        )

def _calculate_seq_nlls(
    data_dir: str,
    model,
    vocab,
    family: str,
    device: str = 'cuda',
    batch_size: int = 32,
    output_dir: str = None
):
    output_path = os.path.join(output_dir, family + ".txt")
    ds = PeintDataset(
        data_path = data_dir,
        vocab = vocab,
        max_len = 1022,
        families = [family]
    )

    collator = PeintCollator(
        vocab = vocab,
        mask_prob = 0,  # don't mask x
    )

    all_nlls = []
    # shuffling = False to ensure sequences are processed in the same order independent of whether or not they're reestimated
    data_loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator, shuffle=False) 

    for batch in data_loader:    
        batched = [b.to(device) for b in batch]

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                x, x_t, y, y_t, t, xmask, ymask = batched
                x_logits, y_logits = model(x, y, t, xmask, ymask)

                nlls = F.cross_entropy(
                    y_logits.transpose(-1,-2), y_t, reduction='none'
                ).detach().cpu().numpy() # this is for ONE sequence and it returns per-site NLL's
                
                mask = (y_t != vocab.padding_idx).detach().cpu().numpy() 
                masked_nlls = nlls * mask 
                
                sum_nlls = masked_nlls.sum(axis=1)  
                non_padding_counts = mask.sum(axis=1)
                mean_nlls = sum_nlls / non_padding_counts

        all_nlls.extend(mean_nlls)
    io.write_nlls(nlls=all_nlls, output_path=output_path)
    secure_parallel_output(output_dir=output_dir, parallel_arg=family, suffix='.txt')

def _map_func_calculate_seq_nlls(args):
    data_dir, model, vocab, device, families, batch_size, output_dir = args

    for family in families:
        _calculate_seq_nlls(
            data_dir=data_dir,
            model=model,
            vocab=vocab,
            family=family,
            device=device,
            batch_size=batch_size,
            output_dir=output_dir
        )

@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes", "batch_size", "model", "vocab"],
    exclude_args_if_default=["model_name"],
    output_dirs=["output_dir"]
)
def calculate_seq_nlls(
    data_dir: str,
    model,
    vocab,
    device: str = 'cuda',
    families: List[str] = [],
    batch_size: int = 32,
    num_processes: int = 1,
    model_name: str = None,
    output_dir: Optional[str] = None
):
    logger = logging.getLogger(__name__)
    logger.info(f"Going to compute sequence NLLs for {len(families)} families")

    if not os.path.exists(data_dir):
        raise ValueError(f"Could not find data_dir {data_dir}")

    map_args = [
        [
            data_dir,
            model,
            vocab,
            device,
            get_process_args(process_rank, num_processes, families),
            batch_size,
            output_dir
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_map_func_calculate_seq_nlls, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_map_func_calculate_seq_nlls, map_args),
                total=len(map_args),
            )
        )

def convert_batched_tokens_to_sequences(x, vocab, aa_only=True):
    idx_to_tok = {idx: tok for tok, idx in vocab.tok_to_idx.items()}
    
    sequences = []
    for sequence in x:
        seq = [idx_to_tok[idx.item()] for idx in sequence]
        if aa_only:
            seq = [char for char in seq if char in amino_acids]
        sequences.append("".join(seq))
    
    return sequences
