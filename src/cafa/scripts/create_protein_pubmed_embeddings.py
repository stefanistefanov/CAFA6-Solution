import glob
import logging
import os

import pandas as pd
import torch
from datasets import load_from_disk
from jsonargparse import CLI
from safetensors.torch import load_file, save_file
from torch import Tensor
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def get_protein_pubmed_mappings(
    idmapping_path: str, dataset_dir: str, top_k: int
) -> pd.DataFrame:
    idmap_df = pd.read_csv(idmapping_path, sep='\t')
    idmap_df.fillna({'PubMed': '', 'Additional_PubMed': ''}, inplace=True)
    protein_pubmed_tuples = []
    for protein_rec in idmap_df.to_dict(orient='records'):
        protein_pubmed_ids = protein_rec['PubMed'].split('; ')
        protein_additional_pubmed_ids = protein_rec['Additional_PubMed'].split(
            '; '
        )
        protein_pubmed_tuples.extend(
            [
                (protein_rec['UniProtKB_AC'], pubmed_id)
                for pubmed_id in protein_pubmed_ids
                if pubmed_id
            ]
        )
        protein_pubmed_tuples.extend(
            [
                (protein_rec['UniProtKB_AC'], pubmed_id)
                for pubmed_id in protein_additional_pubmed_ids
                if pubmed_id
            ]
        )
    protein_pubmed_df = pd.DataFrame(
        protein_pubmed_tuples, columns=['protein_id', 'pubmed_id']
    )
    protein_pubmed_df = protein_pubmed_df.drop_duplicates()
    logger.info(f'Available in mapping:\n{protein_pubmed_df.nunique()}')

    pubmed_ds = load_from_disk(dataset_dir)
    pubmed_df = pubmed_ds.select_columns(
        ['PMID', 'PubDate', 'DateCompleted', 'DateRevised']
    ).to_pandas()
    del pubmed_ds

    pubdate_df = pubmed_df[['PMID', 'PubDate']].copy()
    pubdate_df.rename(columns={'PMID': 'pubmed_id'}, inplace=True)
    pubdate_df['PubDate'] = pd.to_datetime(
        pubdate_df['PubDate'], errors='coerce'
    )
    # Fill missing PubDate with DateCompleted/DateRevised 
    pubdate_df.loc[pubdate_df['PubDate'].isna(), 'PubDate'] = pd.to_datetime(
        pubmed_df.loc[pubdate_df['PubDate'].isna(), 'DateCompleted'],
        errors='coerce',
    )
    pubdate_df.loc[pubdate_df['PubDate'].isna(), 'PubDate'] = pd.to_datetime(
        pubmed_df.loc[pubdate_df['PubDate'].isna(), 'DateRevised'],
        errors='coerce',
    )
    del pubmed_df

    protein_pubmed_df = protein_pubmed_df.merge(
        pubdate_df, on='pubmed_id', how='left'
    )
    logger.info(
        f'Available in downloaded dataset:\n'
        f"{protein_pubmed_df[['protein_id', 'pubmed_id']].nunique()}"
    )

    # Select the top k most recent PubMed IDs for each protein
    protein_pubmed_df = (
        protein_pubmed_df.sort_values(
            ['protein_id', 'PubDate'], ascending=[True, False]
        )
        .groupby('protein_id')
        .head(top_k)
        .reset_index(drop=True)
    )

    return protein_pubmed_df


def load_pubmed_embeddings(
    pubmed_embeddings_dir: str, pubmed_ids: set[str]
) -> dict[str, Tensor]:
    embedding_files = sorted(
        glob.glob(os.path.join(pubmed_embeddings_dir, '*.safetensors'))
    )
    pubmed_embedding_dict = {}
    for emb_file in embedding_files:
        emb_data = load_file(emb_file)
        pubmed_embedding_dict.update(
            {
                pubmed_id: emb
                for pubmed_id, emb in emb_data.items()
                if pubmed_id in pubmed_ids
            }
        )
    return pubmed_embedding_dict


def create_protein_pubmed_embeddings(
    idmapping_path: str,
    dataset_dir: str,
    pubmed_embeddings_dir: str,
    output_embeddings_path: str,
    top_k: int,
) -> None:
    assert not os.path.exists(
        output_embeddings_path
    ), f'Output file {output_embeddings_path} already exists.'
    os.makedirs(os.path.dirname(output_embeddings_path), exist_ok=True)

    protein_pubmed_df = get_protein_pubmed_mappings(
        idmapping_path, dataset_dir, top_k
    )
    logger.info(
        f'For embeddings creation:\n'
        f"{protein_pubmed_df[['protein_id', 'pubmed_id']].nunique()}"
    )
    pubmed_embedding_dict = load_pubmed_embeddings(
        pubmed_embeddings_dir, pubmed_ids=set(protein_pubmed_df['pubmed_id'])
    )
    logger.info(f'Loaded {len(pubmed_embedding_dict)} PubMed embeddings')

    protein_pubmed_embeddings = {}
    for protein_id, group in tqdm(protein_pubmed_df.groupby('protein_id')):
        embeddings = []
        for pubmed_id in group['pubmed_id']:
            if pubmed_id in pubmed_embedding_dict:
                embeddings.append(pubmed_embedding_dict[pubmed_id])
        if embeddings:
            # Compute mean and max of all available embeddings for this protein
            embeddings = torch.stack(embeddings)
            embeddings = torch.cat(
                (embeddings.mean(dim=0), embeddings.max(dim=0).values)
            )
            protein_pubmed_embeddings[protein_id] = embeddings

    save_file(protein_pubmed_embeddings, output_embeddings_path)
    logger.info(
        f'Saved {len(protein_pubmed_embeddings)} protein PubMed embeddings to '
        f'{output_embeddings_path}'
    )


if __name__ == '__main__':
    CLI(create_protein_pubmed_embeddings, as_positional=False)
