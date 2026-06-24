from enum import Enum
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import obonet
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

MAX_TERMS = 500


class Subontology(Enum):
    BP = 'biological_process'
    CC = 'cellular_component'
    MF = 'molecular_function'


SUBONTOLOGY_TO_ASPECT = {
    Subontology.BP: 'P',
    Subontology.CC: 'C',
    Subontology.MF: 'F',
}


class CAFAMetrics:
    def __init__(
        self,
        data_dir: Path,
        gt_terms_df: pd.DataFrame,
        propagate_gt: bool,
        exclude_terms_df: pd.DataFrame = None,
    ):
        """
        :param exclude_terms_df: known terms to be excluded from evaluation. They will be
        propagated if propagate_gt is True. They will be excluded from both GT and predictions.
        """
        go_graph = obonet.read_obo(
            data_dir / 'Train/go-basic.obo', ignore_obsolete=True
        )
        self.initialize_subontology_graphs(go_graph)
        self.term_to_index_per_subontology: Dict[
            Subontology, Dict[str, int]
        ] = {
            subontology: {
                term: term_idx
                for term_idx, term in enumerate(
                    sorted(list(subontology_graph.nodes))
                )
            }
            for subontology, subontology_graph in self.subontology_graphs.items()
        }

        # assert that terms are sorted as expected
        for term_to_index in self.term_to_index_per_subontology.values():
            assert list(term_to_index.keys()) == sorted(term_to_index.keys())

        self.term_to_subontology = {
            term: subontology
            for subontology, term_to_index in self.term_to_index_per_subontology.items()
            for term in term_to_index.keys()
        }
        self.initialize_term_to_predcessor_indices()

        ia_df = pd.read_csv(
            data_dir / 'IA.tsv',
            sep='\t',
            header=None,
            index_col=None,
            names=['term', 'weight'],
        )

        # GO terms are ordered in the same way as in Information accretion data
        assert (
            ia_df['term'].values
            == np.array(list(self.term_to_subontology.keys()))
        ).all()
        term_to_weight = dict(zip(ia_df['term'], ia_df['weight']))
        self.ia_arr_per_subontology = {
            subontology: np.array(
                [term_to_weight[term] for term in term_to_index.keys()]
            )
            for subontology, term_to_index in self.term_to_index_per_subontology.items()
        }
        self.initialize_gt(gt_terms_df, propagate_gt, exclude_terms_df)

    def initialize_subontology_graphs(self, go_graph):
        filter_edges(go_graph)
        subontology_graphs = {}
        for cc in nx.weakly_connected_components(go_graph):
            subontology_graph = go_graph.subgraph(cc)
            node = next(iter(subontology_graph.nodes()))
            node_attributes = subontology_graph.nodes[node]
            subontology_graph.name = node_attributes['namespace']
            assert Subontology(subontology_graph.name) in Subontology
            assert all(
                subontology_graph.name == node_namespace
                for _, node_namespace in subontology_graph.nodes(
                    data='namespace'
                )
            )
            subontology_graphs[Subontology(subontology_graph.name)] = (
                subontology_graph
            )

        # Order subontology_graphs in the same order as in Subontology enum
        self.subontology_graphs = {
            subontology: subontology_graphs[subontology]
            for subontology in Subontology
        }

        self.subontology_edges = {
            # Edge nodes are reversed to be in the expected order
            # the parent is first, child is second
            subontology: [(edge[1], edge[0]) for edge in so_graph.edges]
            for subontology, so_graph in self.subontology_graphs.items()
        }

    def initialize_term_to_predcessor_indices(self):
        self.term_to_predecessor_indices_per_subontology = {}
        for subontology, subontology_graph in self.subontology_graphs.items():
            topologically_sorted_terms = list(
                nx.topological_sort(subontology_graph)
            )
            # remove terms with in_degree==0 they are not needed for propagation
            topologically_sorted_terms = [
                term
                for term in topologically_sorted_terms
                if subontology_graph.in_degree(term) > 0
            ]
            term_to_index = self.term_to_index_per_subontology[subontology]

            # the order of the terms in the dictionary is important as they
            # have to be iterated in this order for correct propagation
            self.term_to_predecessor_indices_per_subontology[subontology] = {
                term: [
                    term_to_index[predecessor_term]
                    for predecessor_term in subontology_graph.predecessors(term)
                ]
                for term in topologically_sorted_terms
            }
            assert (
                list(
                    self.term_to_predecessor_indices_per_subontology[
                        subontology
                    ].keys()
                )
                == topologically_sorted_terms
            )

    def initialize_gt(
        self,
        gt_terms_df: pd.DataFrame,
        propagate_gt: bool,
        exclude_terms_df: pd.DataFrame = None,
    ):
        self.protein_id_to_index = {
            protein_id: idx
            for idx, protein_id in enumerate(
                sorted(gt_terms_df['EntryID'].unique())
            )
        }

        self.gt_per_subontology: dict[Subontology, csr_matrix] = {}
        self.exclude_per_subontology: dict[Subontology, csr_matrix] = (
            {} if exclude_terms_df is not None else None
        )
        for subontology in Subontology:
            # all subontology terms in sorted order
            all_subontology_terms = list(
                self.term_to_index_per_subontology[subontology].keys()
            )
            binarizer = MultiLabelBinarizer(
                classes=all_subontology_terms, sparse_output=True
            )
            binarizer.fit(None)
            assert (binarizer.classes_ == np.array(all_subontology_terms)).all()

            self.gt_per_subontology[subontology] = (
                self.create_protein_term_matrix(
                    gt_terms_df, subontology, binarizer
                )
            )
            if exclude_terms_df is not None:
                self.exclude_per_subontology[subontology] = (
                    self.create_protein_term_matrix(
                        exclude_terms_df, subontology, binarizer
                    )
                )

        if propagate_gt:
            self.propagate(self.gt_per_subontology, use_dense=True)
            if self.exclude_per_subontology is not None:
                self.propagate(self.exclude_per_subontology, use_dense=True)

    def create_protein_term_matrix(
        self,
        terms_df: pd.DataFrame,
        subontology: Subontology,
        binarizer: MultiLabelBinarizer,
    ) -> csr_matrix:
        protein_terms = terms_df.loc[
            terms_df['aspect'] == SUBONTOLOGY_TO_ASPECT[subontology],
            ['EntryID', 'term'],
        ]
        protein_terms = protein_terms.groupby(['EntryID'])['term'].apply(list)
        protein_ids_no_term = set(self.protein_id_to_index.keys()) - set(
            protein_terms.index
        )
        protein_terms = pd.concat(
            [
                protein_terms,
                pd.Series(
                    data=len(protein_ids_no_term) * [[]],
                    index=protein_ids_no_term,
                ),
            ]
        )
        protein_terms = protein_terms.sort_index()
        assert list(self.protein_id_to_index) == protein_terms.index.tolist()
        protein_term_matrix = binarizer.transform(protein_terms)
        assert isinstance(protein_term_matrix, csr_matrix)
        protein_term_matrix = protein_term_matrix.astype(np.bool_)
        return protein_term_matrix

    def to_full_prediction_probs_per_subontology(
        self,
        prediction_probs: np.ndarray,
        # correspond to columns in prediction_probs
        prediction_terms: List[str],
        propagate: bool = True,
    ) -> Dict[Subontology, np.ndarray]:
        num_rows = prediction_probs.shape[0]
        full_prediction_probs_per_subontology: Dict[Subontology, np.ndarray] = (
            {}
        )
        for subontology in Subontology:
            subontology_pred_indices = [
                pred_idx
                for pred_idx, term in enumerate(prediction_terms)
                if self.term_to_subontology[term] == subontology
            ]
            subontology_pred_terms = [
                prediction_terms[pred_idx]
                for pred_idx in subontology_pred_indices
            ]
            subontology_term_to_index = self.term_to_index_per_subontology[
                subontology
            ]
            full_subontology_pred_indices = [
                subontology_term_to_index[so_term]
                for so_term in subontology_pred_terms
            ]
            num_cols = len(subontology_term_to_index)
            full_subontology_pred_probs = np.zeros(
                (num_rows, num_cols), dtype=prediction_probs.dtype
            )
            full_subontology_pred_probs[:, full_subontology_pred_indices] = (
                prediction_probs[:, subontology_pred_indices]
            )

            if propagate:
                self.propagate_subontology(
                    full_subontology_pred_probs, subontology
                )
            full_prediction_probs_per_subontology[subontology] = (
                full_subontology_pred_probs
            )
        return full_prediction_probs_per_subontology

    def take_max_terms_per_subontology(
        self,
        full_prediction_probs_per_subontology: Dict[Subontology, np.ndarray],
    ) -> Dict[Subontology, csr_matrix]:
        prediction_probs_per_subontology: Dict[Subontology, csr_matrix] = {}
        for subontology in Subontology:
            full_subontology_pred_probs = full_prediction_probs_per_subontology[
                subontology
            ]
            num_rows, num_cols = full_subontology_pred_probs.shape

            topk_prediction_indices = np.argsort(
                full_subontology_pred_probs, axis=1
            )[:, -MAX_TERMS:][:, ::-1]

            assert topk_prediction_indices.shape[1] <= MAX_TERMS, subontology
            topk_prediction_probs = np.take_along_axis(
                full_subontology_pred_probs, topk_prediction_indices, axis=1
            )
            assert topk_prediction_probs.shape[1] <= MAX_TERMS, subontology

            row_indices = np.repeat(
                np.arange(num_rows),
                topk_prediction_indices.shape[1],
            )
            col_indices = topk_prediction_indices.ravel()
            csr_pred_probs = csr_matrix(
                (topk_prediction_probs.ravel(), (row_indices, col_indices)),
                shape=(num_rows, num_cols),
            )
            prediction_probs_per_subontology[subontology] = csr_pred_probs
        return prediction_probs_per_subontology

    def to_prediction_probs_per_subontology(
        self,
        prediction_probs: np.ndarray,
        # correspond to columns in prediction_probs
        prediction_terms: List[str],
        propagate: bool = True,
    ) -> Dict[Subontology, csr_matrix]:
        # prediction_probs should be split first into the three subontologies
        # because MAX_TERMS are taken from each subontology
        full_prediction_probs_per_subontology = (
            self.to_full_prediction_probs_per_subontology(
                prediction_probs, prediction_terms, propagate
            )
        )
        prediction_probs_per_subontology = self.take_max_terms_per_subontology(
            full_prediction_probs_per_subontology
        )
        return prediction_probs_per_subontology

    def propagate(self, prediction_probs_per_subontology, use_dense):
        for subontology in Subontology:
            pred_probs = prediction_probs_per_subontology[subontology]
            if use_dense:
                pred_probs = pred_probs.toarray()
            self.propagate_subontology(pred_probs, subontology)
            prediction_probs_per_subontology[subontology] = (
                csr_matrix(pred_probs) if use_dense else pred_probs
            )

    def propagate_subontology(self, subontology_pred_probs, subontology):
        term_to_index = self.term_to_index_per_subontology[subontology]
        term_to_predecessors_indices = (
            self.term_to_predecessor_indices_per_subontology[subontology]
        )
        for (
            term,
            predecessors_indices,
        ) in term_to_predecessors_indices.items():
            term_idx = term_to_index[term]
            predecessors_max = subontology_pred_probs[
                :, predecessors_indices + [term_idx]
            ].max(axis=1)
            subontology_pred_probs[:, term_idx] = predecessors_max

    def to_protein_term_prediction_probs(
        self,
        prediction_probs_per_subontology: Dict[Subontology, csr_matrix],
        prediction_protein_ids: List[str],
    ):
        index_to_term_per_subontology = {
            subontology: {index: term for term, index in term_to_index.items()}
            for subontology, term_to_index in (
                self.term_to_index_per_subontology.items()
            )
        }
        pred_protein_ids = []
        pred_probs = []
        pred_terms = []
        for protein_index, protein_id in enumerate(prediction_protein_ids):
            for subontology in Subontology:
                index_to_term = index_to_term_per_subontology[subontology]
                protein_subontology_pred_probs = (
                    prediction_probs_per_subontology[subontology][protein_index]
                )
                nonzero_indices = protein_subontology_pred_probs.nonzero()
                _, term_indices = nonzero_indices
                if len(term_indices) == 0:
                    continue
                protein_pred_probs = protein_subontology_pred_probs[
                    nonzero_indices
                ]
                # The score must be in the interval (0, 1.000] and contain up
                # to 3 (three) significant figures.
                protein_pred_probs = protein_pred_probs.getA1()
                nonzero_probs_mask = protein_pred_probs >= 0.001
                protein_pred_probs = protein_pred_probs[nonzero_probs_mask]
                protein_pred_probs = protein_pred_probs.round(3).astype(
                    np.float32
                )
                terms = [
                    index_to_term[term_index]
                    for term_index, nonzero_prob in zip(
                        term_indices, nonzero_probs_mask
                    )
                    if nonzero_prob
                ]
                pred_probs.append(protein_pred_probs)
                pred_protein_ids.extend(len(protein_pred_probs) * [protein_id])
                pred_terms.extend(terms)
        pred_probs = np.concatenate(pred_probs)
        protein_term_pred_probs_df = pd.DataFrame(
            {
                'protein_id': pred_protein_ids,
                'term': pred_terms,
                'prob': pred_probs,
            }
        )
        protein_term_pred_probs_df = protein_term_pred_probs_df.sort_values(
            by=['protein_id', 'prob', 'term'], ascending=[True, False, True]
        )
        return protein_term_pred_probs_df

    def compute(
        self,
        prediction_probs_per_subontology: Dict[Subontology, csr_matrix],
        # correspond to rows in prediction_probs
        prediction_protein_ids: List[str],
        # thresholds
        tau_arr=np.arange(0.01, 1, 0.01),
        normalization='cafa',
    ):
        gt_protein_indices = [
            self.protein_id_to_index[protein_id]
            for protein_id in prediction_protein_ids
        ]
        subontology_metrics_dfs = []
        for subontology in Subontology:
            ia_arr = self.ia_arr_per_subontology[subontology]
            pred = prediction_probs_per_subontology[subontology]
            gt = self.gt_per_subontology[subontology][gt_protein_indices]
            exclude = (
                self.exclude_per_subontology[subontology][gt_protein_indices]
                if self.exclude_per_subontology is not None
                else None
            )
            metrics_df = self.compute_subontology(
                gt,
                pred,
                exclude,
                ia_arr,
                subontology.value,
                tau_arr,
                normalization,
            )
            subontology_metrics_dfs.append(metrics_df)
        metrics_df = pd.concat(subontology_metrics_dfs)
        metrics_df = metrics_df[
            ['ns', 'tau', 'cov', 'pr', 'rc', 'f']
            + ['wpr', 'wrc', 'wf', 'mi', 'ru', 's']
        ]

        df = metrics_df.copy()
        df = df[df['cov'] > 0].reset_index(drop=True)
        df.set_index(['ns', 'tau'], inplace=True)

        max_metric_dfs = {}
        for metric, cols in [
            ('f', ['rc', 'pr']),
            ('wf', ['wrc', 'wpr']),
            ('s', ['ru', 'mi']),
        ]:
            index_best = (
                df.groupby(level=['ns'])[metric].idxmax()
                if metric in ['f', 'wf']
                else df.groupby(['ns'])[metric].idxmin()
            )
            df_best = df.loc[index_best]
            df_best['max_cov'] = (
                df.reset_index('tau').groupby(level=['ns'])['cov'].max().values
            )
            max_metric_dfs[metric] = df_best.reset_index()
        return metrics_df, max_metric_dfs

    def compute_per_protein(
        self,
        prediction_probs_per_subontology: Dict[Subontology, csr_matrix],
        prediction_protein_ids: List[str],
        tau_per_subontology: Dict[Subontology, float],
        normalization='cafa',
    ):
        tau_arr_per_subontology: Dict[Subontology, np.ndarray] = {
            so: np.array([tau]) for so, tau in tau_per_subontology.items()
        }
        all_protein_ids = []
        all_metrics_df = []
        for (
            subontology,
            prediction_probs_so,
        ) in prediction_probs_per_subontology.items():
            ia_arr = self.ia_arr_per_subontology[subontology]
            tau_arr = tau_arr_per_subontology[subontology]
            for pred_protein_index, protein_id in tqdm(
                enumerate(prediction_protein_ids),
                total=len(prediction_protein_ids),
            ):
                pred = prediction_probs_so[pred_protein_index]
                gt_protein_index = self.protein_id_to_index[protein_id]
                gt = self.gt_per_subontology[subontology][gt_protein_index]
                if gt.count_nonzero() == 0:
                    continue
                exclude = (
                    self.exclude_per_subontology[subontology][gt_protein_index]
                    if self.exclude_per_subontology is not None
                    else None
                )
                so_metrics_df = self.compute_subontology(
                    gt,
                    pred,
                    exclude,
                    ia_arr,
                    subontology.value,
                    tau_arr,
                    normalization,
                )
                all_metrics_df.append(so_metrics_df)
                all_protein_ids.append(protein_id)
        all_metrics_df = pd.concat(all_metrics_df)
        all_metrics_df['protein_id'] = all_protein_ids
        return all_metrics_df

    def compute_subontology(
        self,
        gt: csr_matrix,
        pred: csr_matrix,
        exclude: csr_matrix | None,
        ia_arr: np.ndarray,
        subontology_full_name: str,
        tau_arr: np.ndarray,
        normalization: str,
    ):
        if exclude is not None:
            assert gt.shape == pred.shape == exclude.shape, (
                gt.shape,
                pred.shape,
                exclude.shape,
            )
            # Remove excluded terms from both GT and predictions keeping sparsity.
            gt = gt - gt.multiply(exclude)
            pred = pred - pred.multiply(exclude)

        gt_present = np.asarray(gt.sum(axis=1)).squeeze(1) > 0
        gt = gt[gt_present]
        pred = pred[gt_present]
        assert gt.shape == pred.shape, (gt.shape, pred.shape)

        # terms of interest
        toi = np.nonzero(ia_arr > 0)[0]
        g = gt[:, toi]
        n_gt = np.asarray(g.sum(axis=1))
        ia_toi_csr = csr_matrix(ia_arr[toi])
        wn_gt = np.asarray((g.multiply(ia_toi_csr)).sum(axis=1))
        # cov - coverage, ru - remaining uncertainty, mi - misinformation
        # cov, pr, rc, wpr, wrc, ru, mi
        metrics = np.zeros((len(tau_arr), 7), dtype='float')
        for i, tau in enumerate(tau_arr):
            p = pred[:, toi] >= tau
            # number of proteins with at least one term predicted with
            # score >= tau - coverage
            metrics[i, 0] = (p.sum(axis=1) > 0).sum()

            # Terms subsets; TP
            intersection = g.multiply(p)

            # Subsets size
            n_pred = np.asarray(p.sum(axis=1))
            n_intersection = np.asarray(intersection.sum(axis=1))

            # Precision, recall
            metrics[i, 1] = np.divide(
                n_intersection,
                n_pred,
                out=np.zeros_like(n_intersection, dtype='float'),
                where=n_pred > 0,
            ).sum()
            metrics[i, 2] = np.divide(
                n_intersection,
                n_gt,
                out=np.zeros_like(n_gt, dtype='float'),
                where=n_gt > 0,
            ).sum()

            # Terms subsets
            # FN --> not predicted but in the ground truth
            # remaining = (p != g).multiply(g)
            # FP --> predicted but not in the ground truth
            # mis = (p != g).multiply(p)
            # FN --> not predicted but in the ground truth
            remaining = p < g
            # FP --> predicted but not in the ground truth
            mis = p > g

            wn_pred = np.asarray(p.multiply(ia_toi_csr).sum(axis=1))
            wn_intersection = np.asarray(
                intersection.multiply(ia_toi_csr).sum(axis=1)
            )

            metrics[i, 3] = np.divide(
                wn_intersection,
                wn_pred,
                out=np.zeros_like(n_intersection, dtype='float'),
                where=n_pred > 0,
            ).sum()
            metrics[i, 4] = np.divide(
                wn_intersection,
                wn_gt,
                out=np.zeros_like(n_intersection, dtype='float'),
                where=n_gt > 0,
            ).sum()

            # Misinformation, remaining uncertainty
            metrics[i, 5] = remaining.multiply(ia_toi_csr).sum(axis=1).sum()
            metrics[i, 6] = mis.multiply(ia_toi_csr).sum(axis=1).sum()
        metrics = pd.DataFrame(
            metrics, columns=["cov", "pr", "rc", "wpr", "wrc", "ru", "mi"]
        )
        ne = np.full(len(tau_arr), gt.shape[0])
        for column in ["pr", "rc", "wpr", "wrc", "ru", "mi"]:
            if normalization == 'gt' or (
                column in ["rc", "wrc", "ru", "mi"] and normalization == 'cafa'
            ):
                metrics[column] = np.divide(
                    metrics[column],
                    ne,
                    out=np.zeros_like(metrics[column], dtype='float'),
                    where=ne > 0,
                )
            else:
                metrics[column] = np.divide(
                    metrics[column],
                    metrics["cov"],
                    out=np.zeros_like(metrics[column], dtype='float'),
                    where=metrics["cov"] > 0,
                )
        metrics['ns'] = [subontology_full_name] * len(tau_arr)
        metrics['tau'] = tau_arr
        metrics['cov'] = np.divide(
            metrics['cov'],
            ne,
            out=np.zeros_like(metrics['cov'], dtype='float'),
            where=ne > 0,
        )
        metrics['f'] = compute_f(metrics['pr'], metrics['rc'])
        metrics['wf'] = compute_f(metrics['wpr'], metrics['wrc'])
        metrics['s'] = compute_s(metrics['ru'], metrics['mi'])
        return metrics


def compute_f(pr, rc):
    n = 2 * pr * rc
    d = pr + rc
    return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0)


def compute_s(ru, mi):
    return np.sqrt(ru**2 + mi**2)

def filter_edges(go_graph, valid_edges=None):
    """
    Keep only is_a, part_of edges as in CAFA evaluator obo_parser() these
    are the edges that are used for propagation.
    """
    if valid_edges is None:
        valid_edges = {'is_a', 'part_of'}
    edges_to_remove = []
    for edge in go_graph.edges:
        edge_name = edge[2]
        if edge_name not in valid_edges:
            edges_to_remove.append(edge)
    go_graph.remove_edges_from(edges_to_remove)


def get_max_metrics(max_metric_dfs, max_metrics_key) -> Dict[str, float]:
    metrics = max_metric_dfs[max_metrics_key]
    metrics = metrics.set_index('ns').stack().reset_index()
    metrics.columns = ['ns', 'metric', 'value']
    metrics['metric'] = (
        f'max_{max_metrics_key}/' + metrics['ns'] + '/' + metrics['metric']
    )
    metrics = (
        metrics[['metric', 'value']].set_index('metric')['value'].to_dict()
    )
    return metrics
