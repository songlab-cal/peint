import os

SecondaryStructureType = str


def read_secondary_structure(
    secondary_structure_path: str,
) -> SecondaryStructureType:
    res = open(secondary_structure_path, "r").read()
    return res


def write_secondary_structure(
    secondary_structure: SecondaryStructureType, secondary_structure_path: str
) -> None:
    secondary_structure_dir = os.path.dirname(secondary_structure_path)
    if not os.path.exists(secondary_structure_dir):
        os.makedirs(secondary_structure_dir)
    with open(secondary_structure_path, "w") as outfile:
        outfile.write(secondary_structure)
        outfile.flush()
