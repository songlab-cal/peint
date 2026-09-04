from typing import List, Tuple, Optional, Union, Dict

import numpy as np
import pandas as pd
from cherryml import markov_chain
from Bio import SeqIO

amino_acids = (
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
)

secondary_structure_annotations = [
    'H',
    'B',
    'E',
    'G',
    'I',
    'T',
    'S',
]

gap_character = "-"

ambiguous_mapping = {
    'B': 'N',  
    'Z': 'Q',  
    'J': 'I',  
    'U': 'C',  # Selenocysteine -> Cysteine (closest)
    'O': 'K',  # Pyrrolysine -> Lysine (closest)
}

minimum_distance_for_nontrivial_contact = 7

filter_function_str_dict = {
    "no stratifying (default)": "lambda x, y, t: [i for i in range(len(y))]",
    "x_i: gap, y_i: gap": "lambda x, y, t: [i for i in range(len(y)) if x[i] == '-' and y[i] == '-']",
    "x_i: no gap, y_i: gap": "lambda x, y, t: [i for i in range(len(y)) if x[i] != '-' and y[i] == '-']",
    "x_i: gap, y_i: no gap": "lambda x, y, t: [i for i in range(len(y)) if x[i] == '-' and y[i] != '-']",
    "x_i: no gap, y_i: no gap": "lambda x, y, t: [i for i in range(len(y)) if x[i] != '-' and y[i] != '-']",
    "y_i: gap": "lambda x, y, t: [i for i in range(len(y)) if y[i] == '-']",
    "y_i: no gap": "lambda x, y, t: [i for i in range(len(y)) if y[i] != '-']",
}


def get_process_args(
    process_rank: int, num_processes: int, all_args: List
) -> List:
    """
    Helper function for sharding when using multiprocessing.
    """
    process_args = [
        all_args[i]
        for i in range(len(all_args))
        if i % num_processes == process_rank
    ]
    return process_args


def get_quantization_points_from_geometric_grid(
    quantization_grid_center: float = 0.03,
    quantization_grid_step: float = 1.1,
    quantization_grid_num_steps: int = 64,
) -> List[str]:
    quantization_points = [
        ("%.8f" % (quantization_grid_center * quantization_grid_step**i))
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]
    return quantization_points


PLT_SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "dpi": 300,
}


def matrix_exponential_reversible(
    rate_matrix: np.array,
    exponents: List[float],
) -> np.array:
    """
    Compute matrix exponential (batched).

    Args:
        rate_matrix: Rate matrix for which to compute the matrix exponential
        exponents: List of exponents.
    Returns:
        3D tensor where res[:, i, i] contains exp(rate_matrix * exponents[i])
    """
    return markov_chain.matrix_exponential_reversible(
        exponents=exponents,
        fact=markov_chain.FactorizedReversibleModel(rate_matrix),
        device="cpu",
    )


def compute_quantiles(
    transitions: List[Tuple],
    families: List[str],
    num_quantiles: int,
    pad: float = 0.01,
) -> List[float]:
    """Compute quantiles from transitions for each quantile across all families.
        
    Bins work like histogram bins where the left boundary is inclusive and the 
    right boundary is exclusive. it leaves a special case for the last (very right)
    bin, which should be inclusive of both left and right boundaries. The pad is to 
    make sure that last bin contains the 100th percentile time.

    Bins work like histogram bins where the left boundary is inclusive and the 
    right boundary is exclusive. it leaves a special case for the last (very right)
    bin, which should be inclusive of both left and right boundaries. The pad is to 
    make sure that last bin contains the 100th percentile time.

    Args:
        transitions (List[Tuple]): list of transitions in the form (x, y, t)
        families (List[str]): list of protein families
        num_quantiles (int): number of quantiles
        pad (float, optional): value to pad to the last bucket. Defaults to 0.01.

    Returns:
        List[float] of length num_quantiles+1 with the left and right bounds of the quantiles
    """
    # get all the times
    all_times = [t for (x, y, t) in transitions]
    # sort them
    all_times.sort()

    quantiles = pd.qcut(all_times, num_quantiles, precision=3, retbins=True)[1]
    # left inclusive, right exclusive bins
    quantiles[-1] += pad
    return quantiles


def get_quantile_idx(quantiles: List[float], t: float) -> int:
    """Returns the quantile index that time t falls in.

    Args:
        quantiles (List[float]): List of len(quantiles)-1 quantiles where each quantile is denoted by [quantiles[i], quantiles[i+1]).
        t (float): time t that we want the quantile index of.

    Returns:
        int quantile_idx between [0, len(quantiles)-2] where t falls between quantiles[quantile_idx] and quantiles[quantile_idx+1]. If t is smaller than quantiles[0], it belongs in the first quantile. If t is greater than quantiles[-1], it belongs in the last quantile .
    """
    if t < quantiles[0]:
        return 0
    elif t > quantiles[-1]:
        return len(quantiles) - 2

    idx_to_insert_t = np.searchsorted(quantiles, t, "right")
    return idx_to_insert_t - 1


def one_hot_encode(proteins: Union[str, List[str]], alphabet: List[str]):
    """One-hot-encodes proteins."""
    if type(proteins) == str:
        proteins = [proteins]
    proteins = proteins.copy()
    proteins.append(
        "".join(alphabet)
    )  # make sure all alphabet shows up as columns
    ohe = pd.get_dummies(pd.Series(list("".join(proteins))), dtype=int)
    ohe = ohe[: -len(alphabet)]  # remove last dummy string
    ohe = ohe.reindex(columns=alphabet)
    return ohe

def one_hot_encode_prev(proteins: Union[str, List[str]], alphabet: List[str], suffix: str, ohe: Optional[pd.DataFrame] = None) -> pd.DataFrame: 
    """
    One-hot-encodes x-1 in place of each amino acid x. 
    For amino acids at the start of the protein, pads x-1 with zeros.
    
    Args: 
        proteins (Union[str, List[str]]): a single protein or a list of proteins
        alphabet (List[str]): list of possible amino acids
        suffix (str): suffix to add to result's column names
        ohe (optional pd.DataFrame): include if the OHE has already been computed. 
    """
    # if one hot encoding has been previously calculated, copy. else, recalculate
    if ohe is None: 
        ohe = one_hot_encode(proteins=proteins, alphabet=alphabet)
    else: 
        ohe = ohe.copy() 
        
    X_prev = ohe.shift(periods=1, fill_value=0).add_suffix(suffix)
    
    # zero out x_{i-1} for starting amino acid
    sequence_lengths = pd.Series(proteins).str.len()
    start_amino_acid_index = sequence_lengths.shift(periods=1, fill_value=0).cumsum()
    X_prev.iloc[start_amino_acid_index, :] = np.zeros(len(alphabet))
    
    return X_prev

def one_hot_encode_next(proteins: Union[str, List[str]], alphabet: List[str], suffix: str, ohe: Optional[pd.DataFrame] = None): 
    """
    One-hot-encodes x+1 in place of each amino acid x. 
    For amino acids at the end of the protein, pads x+1 with zeros.
    
    Args: 
        proteins (Union[str, List[str]]): a single protein or a list of proteins
        alphabet (List[str]): list of possible amino acids
        suffix (str): suffix to add to result's column names
        ohe (optional pd.DataFrame): include if the OHE has already been computed. 
    """
    # if one hot encoding has been previously calculated, copy. else, recalculate
    if ohe is None: 
        ohe = one_hot_encode(proteins=proteins, alphabet=alphabet)
    else: 
        ohe = ohe.copy() 
        
    X_next = ohe.shift(periods=-1, fill_value=0).add_suffix(suffix)
    
    # zero out x_{i+1} for ending amino acid
    sequence_lengths = pd.Series(proteins).str.len()
    end_amino_acid_index = sequence_lengths.cumsum() - 1
    X_next.iloc[end_amino_acid_index, :] = np.zeros(len(alphabet))
    
    return X_next

def _fill_gaps(forward: bool = True): 
    """Helper function for forward_fill_gaps and backward_fill_gaps.
    
    Returns a function that takes in a string representing a protein that forward fills
    (when forward=True) or backward fills the gap characters. 
    """
    def fill_gaps(protein: str): 
        chars = pd.Series(list("".join(protein))) # change string to pd.Series of characters
        chars = chars.replace(gap_character, pd.NA) # turn gap chars into NaN values
        if forward: 
            chars = chars.ffill() # forward fill NaNs
        else: 
            chars = chars.bfill() # backward fill NaNs
        chars_list = chars.fillna(gap_character).to_list() # turn beginning/ending NaNs back into gap characters
        return ''.join(chars_list) # concat pd.Series of characters back into protein string
    return fill_gaps
    
def forward_fill_gaps(proteins: List[str]) -> List[str]: 
    """Forward fills gap characters. 
    
    For example, an input of ['--abc---', 'a-b-c'] will return 
    ['--abcccc', 'aabbc'].
    """
    proteins = pd.Series(proteins)
    filled = proteins.apply(_fill_gaps(forward=True))
    return filled.to_list()
    
def backward_fill_gaps(proteins: List[str]) -> List[str]: 
    """Backward fills gap characters.
    
    For example, an input of ['--abc---', 'a-b-c']
    will result in ['aaabc---', 'abbcc'].
    """
    proteins = pd.Series(proteins)
    filled = proteins.apply(_fill_gaps(forward=False))
    return filled.to_list()

def stratify_gaps(
    x: str, 
    y: str, 
    t: float, 
    x_gap: Optional[bool] = None, 
    y_gap: Optional[bool] = None,
) -> List[int]:
    """
    Stratifies (x, y, t) protein pairs based on x_gap, y_gap specifications. 
    Returns the log likelihoods that correspond to the filtered x and y amino acids. 
    
    Example: 
        x = 'abcdef--'
        y = '-ab-cd-e'
        t = 0.01
        x_gap=False, 
        y_gap=False
    returns [1,2,4,5] 

    Args:
        x (str): x protein
        y (str): y protein
        log_likelihoods (List[str]): pre-calculated log likelihoods
        x_gap (Optional[bool]): 
            - x_gap=None: both gap and non-gap characters allowed for x
            - x_gap=True: only gap characters allowed for x
            - x_gap=False: only non-gap characters allowed for x
        y_gap (Optional[bool]): 
            - y_gap=None: both gap and non-gap characters allowed for y
            - y_gap=True: only gap characters allowed for y
            - y_gap=False: only non-gap characters allowed for y
    """
    df = pd.DataFrame({'x': list(x), 'y': list(y)})
    
    if x_gap is True: 
        df = df[df['x'] == gap_character]
    elif x_gap is False: 
        df = df[df['x'] != gap_character]
    
    if y_gap is True: 
        df = df[df['y'] == gap_character]
    elif y_gap is False: 
        df = df[df['y'] != gap_character]
        
    return df.index.tolist()

def stratify_amino_acids(
    x: str, 
    y: str, 
    t: float, 
    x_amino_acid: Optional[str] = None, 
    y_amino_acid: Optional[str] = None,
) -> List[int]:
    """
    Stratifies (x, y, t) protein pairs per specified x_amino_acid and y_amino_acid. 
    Returns the log likelihoods that correspond to the filtered x and y amino acids. 
    
    Example: 
        x = 'abcdef--'
        y = '-ab-bb-e'
        t = 0.01
        x_amino_acid = None
        y_amino_acid = 'b'
    returns [2,4,5] 

    Args:
        x (str): x protein
        y (str): y protein
        log_likelihoods (List[str]): pre-calculated log likelihoods
        x_amino_acid (Optional[str]): amino acid to allow for x. If x_amino_acid=None:, 
                                      all amino acid characters allowed for x.
        y_amino_acid (Optional[str]): amino acid to allow for y. If y_amino_acid=None, 
                                      all amino acid characters allowed for y.
    """
    df = pd.DataFrame({'x': list(x), 'y': list(y)})
    
    if x_amino_acid is not None: 
        df = df[df['x'] == x_amino_acid]
    if y_amino_acid is not None: 
        df = df[df['y'] == y_amino_acid]
    
    return df.index.tolist()    
    
def df_to_sparse_matrix(df: pd.DataFrame):
    return df.astype(pd.SparseDtype("float64",0)).sparse.to_coo().tocsr()

def read_msa(msa_path):
    it = SeqIO.parse(msa_path, format='fasta')
    return {record.id: str(record.seq) for record in it}

def write_msa(
    msa: Dict[str, str],
    output_path: str
):
    with open(output_path, 'w') as f:
        for id, seq in msa.items():
            f.write(f">{id}\n{seq}\n")