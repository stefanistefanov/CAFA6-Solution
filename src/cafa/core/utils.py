def extract_protein_id_from_fasta(seq_id: str) -> str:
    """
    Extract protein ID from competition fasta SeqRecord.id.
    May be different formats in train and test data.

    Train Format: sp|{protein_id}|...
    Test Format: {protein_id}
    """
    # Remove '>' if present
    if '|' not in seq_id:
        return seq_id

    seq_id = seq_id.lstrip('>')
    # Split by '|' and extract the protein ID (second element)
    parts = seq_id.split('|')
    if len(parts) >= 2:
        return parts[1]

    raise ValueError(f"Invalid format SeqRecord.id: {seq_id}")
