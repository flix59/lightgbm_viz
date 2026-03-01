"""Extract tree structure and statistics from LightGBM models."""
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any


class TreeExtractor:
    """Extract and process tree information from LightGBM booster."""

    def __init__(self, booster, X_data: pd.DataFrame, y_data: pd.Series = None):
        """
        Initialize tree extractor.

        Args:
            booster: LightGBM booster object
            X_data: Feature data to calculate split statistics
            y_data: Target data (optional, for leaf value analysis)
        """
        self.booster = booster
        self.X_data = X_data
        self.y_data = y_data
        self.feature_names = list(X_data.columns)

        # Get tree structures
        self.tree_info = self._extract_tree_info()
        self._node_indices_cache: Dict[int, Dict[str, np.ndarray]] = {}

    def _extract_tree_info(self) -> List[Dict[str, Any]]:
        """Extract tree structure from booster."""
        model_dump = self.booster.dump_model()
        return model_dump['tree_info']

    def get_num_trees(self) -> int:
        """Get total number of trees in the model."""
        return len(self.tree_info)

    def get_tree(self, tree_idx: int) -> Dict[str, Any]:
        """Get structure of a specific tree."""
        if tree_idx < 0 or tree_idx >= len(self.tree_info):
            raise ValueError(f"Tree index {tree_idx} out of range [0, {len(self.tree_info)})")
        return self.tree_info[tree_idx]['tree_structure']

    def calculate_node_samples(self, tree_idx: int) -> Dict[str, int]:
        """
        Calculate number of samples passing through each node.

        Returns:
            Dictionary mapping node_id -> sample_count
        """
        tree = self.get_tree(tree_idx)
        node_samples = {}

        # Traverse data through tree
        samples = np.arange(len(self.X_data))

        def traverse(node, sample_indices):
            """Recursively traverse tree and count samples at each node."""
            if 'split_index' not in node:
                # Leaf node
                node_samples[f"leaf_{node.get('leaf_index', 'unknown')}"] = len(sample_indices)
                return

            split_feature = node['split_feature']
            threshold = node['threshold']
            decision_type = node.get('decision_type', '<=')

            # Get feature name
            feature_name = self.feature_names[split_feature]

            # Split samples
            feature_values = self.X_data[feature_name].iloc[sample_indices].values

            if decision_type == '<=':
                left_mask = feature_values <= threshold
            else:
                left_mask = feature_values < threshold

            left_indices = sample_indices[left_mask]
            right_indices = sample_indices[~left_mask]

            # Store counts
            node_id = f"split_{node['split_index']}"
            node_samples[node_id] = len(sample_indices)

            # Recurse
            if 'left_child' in node:
                traverse(node['left_child'], left_indices)
            if 'right_child' in node:
                traverse(node['right_child'], right_indices)

        # Start traversal
        traverse(tree, samples)
        return node_samples

    def get_tree_summary(self, tree_idx: int) -> Dict[str, Any]:
        """Get summary statistics for a tree."""
        tree = self.get_tree(tree_idx)
        node_samples = self.calculate_node_samples(tree_idx)

        def count_nodes(node, node_type='all'):
            """Count nodes in tree."""
            if 'split_index' not in node:
                return 1 if node_type in ['all', 'leaf'] else 0
            count = 1 if node_type in ['all', 'split'] else 0
            count += count_nodes(node['left_child'], node_type)
            count += count_nodes(node['right_child'], node_type)
            return count

        total_samples = len(self.X_data)

        return {
            'tree_index': tree_idx,
            'total_nodes': count_nodes(tree, 'all'),
            'split_nodes': count_nodes(tree, 'split'),
            'leaf_nodes': count_nodes(tree, 'leaf'),
            'total_samples': total_samples,
            'node_samples': node_samples
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Calculate feature importance across all trees."""
        importance = self.booster.feature_importance(importance_type='gain')
        feature_names = self.booster.feature_name()

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df

    def get_node_sample_indices(self, tree_idx: int) -> Dict[str, np.ndarray]:
        """
        Get the sample indices that reach each node, cached per tree.

        Returns:
            Dictionary mapping node_id -> array of sample indices
        """
        if tree_idx in self._node_indices_cache:
            return self._node_indices_cache[tree_idx]

        tree = self.get_tree(tree_idx)
        node_indices: Dict[str, np.ndarray] = {}

        def traverse(node, sample_indices: np.ndarray):
            if 'split_index' not in node:
                node_indices[f"leaf_{node.get('leaf_index', 'unknown')}"] = sample_indices
                return

            split_feature = node['split_feature']
            threshold = node['threshold']
            decision_type = node.get('decision_type', '<=')
            feature_name = self.feature_names[split_feature]
            feature_values = self.X_data[feature_name].iloc[sample_indices].values

            left_mask = feature_values <= threshold if decision_type == '<=' else feature_values < threshold

            node_id = f"split_{node['split_index']}"
            node_indices[node_id] = sample_indices

            if 'left_child' in node:
                traverse(node['left_child'], sample_indices[left_mask])
            if 'right_child' in node:
                traverse(node['right_child'], sample_indices[~left_mask])

        traverse(tree, np.arange(len(self.X_data)))
        self._node_indices_cache[tree_idx] = node_indices
        return node_indices

    def get_tree_contributions(self) -> np.ndarray:
        """
        Compute mean absolute contribution of each tree across training data.

        Each tree's raw score is the learning-rate-scaled leaf value applied to
        each sample. The mean absolute value measures how much that tree moves
        predictions on average.

        Returns:
            Array of shape (n_trees,) with mean absolute contributions
        """
        contributions = []
        for i in range(self.get_num_trees()):
            preds = self.booster.predict(self.X_data, start_iteration=i, num_iteration=1)
            contributions.append(float(np.mean(np.abs(preds))))
        return np.array(contributions)

    def predict_leaf_indices(self, tree_idx: int = None) -> np.ndarray:
        """
        Get leaf indices for each sample.

        Args:
            tree_idx: Specific tree index, or None for all trees

        Returns:
            Array of leaf indices (n_samples,) or (n_samples, n_trees)
        """
        if tree_idx is not None:
            return self.booster.predict(self.X_data,
                                       start_iteration=tree_idx,
                                       num_iteration=1,
                                       pred_leaf=True).flatten()
        else:
            return self.booster.predict(self.X_data, pred_leaf=True)
