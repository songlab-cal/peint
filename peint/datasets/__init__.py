"""PEINT dataset components.

Core dataset classes (no Lightning dependency):
    - PeintDataset: PyTorch Dataset for loading protein sequence pairs
    - PeintCollator: Batch collator with MLM masking

Training components (require Lightning, import from peint.datasets.training):
    - PeintDataModule: Lightning DataModule for training
"""

from ._datasets import (
    pfam_15k__treewise_train_test_split__cached,
    a3m_dataset__cached,
    get_a3m_families,
    report_dataset_statistics_str,
    example_pfam_15k__treewise_train_test_split,
)
from ._torch_datasets import (
    PeintDataset,
    PeintCollator,
)

__all__ = [
    # Data processing
    "pfam_15k__treewise_train_test_split__cached",
    "a3m_dataset__cached",
    "get_a3m_families",
    "report_dataset_statistics_str",
    "example_pfam_15k__treewise_train_test_split",
    # PyTorch datasets
    "PeintDataset",
    "PeintCollator",
]
