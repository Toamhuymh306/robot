"""
Module 5: K-means Clustering Classifier
Uses K-means clustering for piece classification (unsupervised learning)
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class KMeansClassifier:
    """
    K-means clustering classifier for tic-tac-toe piece detection.
    Classifies cells into 3 categories: empty, player piece, robot piece.
    
    This is the core unsupervised learning component of the system.
    """
    
    def __init__(self,
                 n_clusters: int = config.N_CLUSTERS,
                 max_iterations: int = config.KMEANS_MAX_ITERATIONS,
                 tolerance: float = config.KMEANS_TOLERANCE,
                 random_state: int = config.KMEANS_RANDOM_STATE):
        """
        Initialize K-means classifier.
        
        Args:
            n_clusters: Number of clusters (3 for tic-tac-toe)
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Create K-means model
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iterations,
            tol=tolerance,
            random_state=random_state,
            n_init=10
        )
        
        # Feature scaler for normalization
        self.scaler = StandardScaler()
        
        # Cluster label mapping (determined after first fit)
        self.cluster_to_label = {}
        self.is_fitted = False
        self.centroids = None
        
    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fit K-means model to feature data.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Cluster labels for each sample
        """
        if features.shape[0] == 0:
            print("[WARNING] No features to fit")
            return np.array([])
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        # Fit K-means
        labels = self.kmeans.fit_predict(features_normalized)
        
        self.centroids = self.kmeans.cluster_centers_
        self.is_fitted = True
        
        return labels
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Predicted cluster labels
        """
        if not self.is_fitted:
            print("[WARNING] Model not fitted, fitting now...")
            return self.fit(features)
        
        features_normalized = self.scaler.transform(features)
        return self.kmeans.predict(features_normalized)
    
    def fit_and_classify(self, features: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Fit model and classify cells, determining which cluster
        corresponds to which piece type.
        
        Args:
            features: Feature matrix of shape (9, n_features) for board cells
            
        Returns:
            Tuple of (board_labels, cluster_info)
            board_labels: Array of [EMPTY, PLAYER, ROBOT] values
        """
        # Fit K-means
        cluster_labels = self.fit(features)
        
        # Analyze clusters to determine mapping
        self.cluster_to_label = self._determine_cluster_mapping(features, cluster_labels)
        
        # Convert cluster labels to board labels
        board_labels = np.array([
            self.cluster_to_label.get(label, config.EMPTY) 
            for label in cluster_labels
        ])
        
        # Prepare cluster info
        cluster_info = {
            'centroids': self.centroids,
            'cluster_to_label': self.cluster_to_label,
            'cluster_counts': {i: np.sum(cluster_labels == i) 
                             for i in range(self.n_clusters)}
        }
        
        return board_labels, cluster_info
    
    def _determine_cluster_mapping(self, 
                                    features: np.ndarray, 
                                    cluster_labels: np.ndarray) -> Dict[int, int]:
        """
        Determine which cluster corresponds to which piece type.
        
        Strategy:
        1. Empty cells typically have the most occurrences (5-7 cells at start)
        2. Empty cells have neutral color values (grayish)
        3. Player and robot pieces have distinct colors
        
        Args:
            features: Original feature vectors
            cluster_labels: K-means assigned labels
            
        Returns:
            Dictionary mapping cluster number to piece type
        """
        n_clusters = self.n_clusters
        cluster_counts = [np.sum(cluster_labels == i) for i in range(n_clusters)]
        cluster_means = []
        
        # Calculate mean features for each cluster
        for i in range(n_clusters):
            mask = cluster_labels == i
            if np.sum(mask) > 0:
                cluster_means.append(np.mean(features[mask], axis=0))
            else:
                cluster_means.append(np.zeros(features.shape[1]))
        
        cluster_means = np.array(cluster_means)
        
        # Assuming HSV features: [H, S, V]
        # Empty cells: Low saturation (S), medium value (V)
        # Colored pieces: High saturation
        
        mapping = {}
        
        # Find empty cluster (highest count or lowest saturation if using HSV)
        empty_cluster = np.argmax(cluster_counts)
        mapping[empty_cluster] = config.EMPTY
        
        # For remaining clusters, use color to distinguish
        remaining = [i for i in range(n_clusters) if i != empty_cluster]
        
        # if len(remaining) >= 2 and features.shape[1] >= 3:
        #     # If using HSV, use Hue to distinguish
        #     # Typical values: Green Hue ~ 60, Red Hue ~ 0 or 170
        #     hue_values = [cluster_means[i][0] if len(cluster_means[i]) > 0 else 0 
        #                  for i in remaining]
            
        #     # Sort by hue value
        #     sorted_remaining = sorted(remaining, key=lambda x: cluster_means[x][0] 
        #                              if len(cluster_means[x]) > 0 else 0)
            
        #     # Lower hue (red-ish) = Robot, Higher hue (green-ish) = Player
        #     # Note: This is a heuristic and may need calibration
        #     if len(sorted_remaining) >= 2:
        #         mapping[sorted_remaining[0]] = config.ROBOT  # Lower hue
        #         mapping[sorted_remaining[1]] = config.PLAYER  # Higher hue
        #     elif len(sorted_remaining) == 1:
        #         mapping[sorted_remaining[0]] = config.PLAYER
        # elif len(remaining) == 1:
        #     mapping[remaining[0]] = config.PLAYER
        if len(remaining) >= 2 and features.shape[1] >= 3:
            hue_0 = cluster_means[remaining[0]][0]
            hue_1 = cluster_means[remaining[1]][0]
            
            for idx in remaining:
                hue = cluster_means[idx][0]
                if hue < 30 or hue > 150:   # Đỏ: Hue ~0 hoặc ~170
                    mapping[idx] = config.ROBOT
                elif 30 <= hue <= 90:        # Xanh lá: Hue ~60
                    mapping[idx] = config.PLAYER
                else:
                    mapping[idx] = config.PLAYER  # Mặc định

        
        return mapping
    
    def classify_single_cell(self, feature: np.ndarray) -> int:
        """
        Classify a single cell.
        
        Args:
            feature: Feature vector for one cell
            
        Returns:
            Piece type (EMPTY, PLAYER, or ROBOT)
        """
        if not self.is_fitted:
            return config.EMPTY
        
        feature = feature.reshape(1, -1)
        feature_normalized = self.scaler.transform(feature)
        cluster = self.kmeans.predict(feature_normalized)[0]
        
        return self.cluster_to_label.get(cluster, config.EMPTY)
    
    def update_cluster_mapping(self, 
                               known_empty_indices: List[int],
                               known_player_indices: List[int],
                               known_robot_indices: List[int],
                               cluster_labels: np.ndarray):
        """
        Update cluster mapping using known labels (semi-supervised).
        
        Args:
            known_empty_indices: Indices of known empty cells
            known_player_indices: Indices of known player cells
            known_robot_indices: Indices of known robot cells
            cluster_labels: Current cluster assignments
        """
        # Count votes for each cluster-label pair
        votes = np.zeros((self.n_clusters, 3))  # 3 classes: empty, player, robot
        
        for idx in known_empty_indices:
            if idx < len(cluster_labels):
                votes[cluster_labels[idx], 0] += 1
        
        for idx in known_player_indices:
            if idx < len(cluster_labels):
                votes[cluster_labels[idx], 1] += 1
        
        for idx in known_robot_indices:
            if idx < len(cluster_labels):
                votes[cluster_labels[idx], 2] += 1
        
        # Assign based on majority vote
        label_map = [config.EMPTY, config.PLAYER, config.ROBOT]
        
        for cluster in range(self.n_clusters):
            best_label = np.argmax(votes[cluster])
            self.cluster_to_label[cluster] = label_map[best_label]
    
    def get_cluster_visualization(self, features: np.ndarray, 
                                   labels: np.ndarray) -> np.ndarray:
        """
        Create a visualization of clusters (2D projection).
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            
        Returns:
            Visualization image
        """
        from sklearn.decomposition import PCA
        
        # Reduce to 2D for visualization
        if features.shape[1] > 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
        else:
            features_2d = features[:, :2]
        
        # Normalize to image coordinates
        min_vals = features_2d.min(axis=0)
        max_vals = features_2d.max(axis=0)
        range_vals = max_vals - min_vals + 1e-6
        
        features_norm = (features_2d - min_vals) / range_vals
        features_pixel = (features_norm * 280 + 10).astype(int)
        
        # Create visualization
        vis = np.full((300, 300, 3), 255, dtype=np.uint8)
        
        # Define colors for clusters
        colors = [
            config.COLOR_EMPTY,
            (0, 255, 0),  # Green for player
            (0, 0, 255)   # Red for robot
        ]
        
        # Draw points
        for i, (point, label) in enumerate(zip(features_pixel, labels)):
            color = colors[label % len(colors)]
            cv2.circle(vis, (point[0], point[1]), 8, color, -1)
            cv2.circle(vis, (point[0], point[1]), 8, (0, 0, 0), 1)
            cv2.putText(vis, str(i), (point[0] - 3, point[1] + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        # Draw centroids if available
        if self.centroids is not None:
            centroids_2d = pca.transform(self.scaler.inverse_transform(self.centroids))
            centroids_norm = (centroids_2d - min_vals) / range_vals
            centroids_pixel = (centroids_norm * 280 + 10).astype(int)
            
            for i, c in enumerate(centroids_pixel):
                cv2.drawMarker(vis, (c[0], c[1]), (0, 0, 0), 
                              cv2.MARKER_CROSS, 20, 2)
        
        return vis


class AdaptiveKMeansClassifier(KMeansClassifier):
    """
    Adaptive K-means that updates centroids incrementally.
    Better for real-time applications where lighting may change.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = 0.1
        self.history_features = []
        self.history_labels = []
        self.max_history = 100
    
    def adaptive_update(self, new_features: np.ndarray, 
                        new_labels: np.ndarray):
        """
        Update model incrementally with new data.
        
        Args:
            new_features: New feature vectors
            new_labels: Known labels for new features
        """
        # Add to history
        self.history_features.extend(new_features.tolist())
        self.history_labels.extend(new_labels.tolist())
        
        # Limit history size
        if len(self.history_features) > self.max_history:
            self.history_features = self.history_features[-self.max_history:]
            self.history_labels = self.history_labels[-self.max_history:]
        
        # Refit with accumulated data
        if len(self.history_features) >= 18:  # At least 2 game's worth
            features = np.array(self.history_features)
            self.fit(features)


def test_kmeans_classifier():
    """
    Test function for K-means classifier module.
    """
    print("=" * 50)
    print("Testing K-means Classifier Module")
    print("=" * 50)
    
    classifier = KMeansClassifier()
    
    # Create synthetic features (simulating HSV mean values)
    # Empty cells: Low saturation, medium value
    empty_features = np.array([
        [90, 30, 150],  # Gray
        [85, 25, 145],
        [95, 35, 155],
        [88, 28, 148],
        [92, 32, 152],
    ])
    
    # Player pieces: Green (high saturation, specific hue)
    player_features = np.array([
        [60, 200, 100],  # Green
        [55, 190, 95],
    ])
    
    # Robot pieces: Red (high saturation, low/high hue)
    robot_features = np.array([
        [5, 220, 120],   # Red
        [175, 210, 115],  # Red (wrap around)
    ])
    
    # Combine features (9 cells: 5 empty, 2 player, 2 robot)
    all_features = np.vstack([empty_features, player_features, robot_features])
    
    print(f"\n[INFO] Feature matrix shape: {all_features.shape}")
    print(f"[INFO] Number of clusters: {classifier.n_clusters}")
    
    # Fit and classify
    board_labels, cluster_info = classifier.fit_and_classify(all_features)
    
    print(f"\n[INFO] Cluster to label mapping: {cluster_info['cluster_to_label']}")
    print(f"[INFO] Cluster counts: {cluster_info['cluster_counts']}")
    
    # Check results
    print("\n[INFO] Classification results:")
    expected = [config.EMPTY] * 5 + [config.PLAYER] * 2 + [config.ROBOT] * 2
    label_names = {config.EMPTY: 'Empty', config.PLAYER: 'Player', config.ROBOT: 'Robot'}
    
    for i, (pred, exp) in enumerate(zip(board_labels, expected)):
        status = "✓" if pred == exp else "✗"
        print(f"  Cell {i}: Predicted={label_names.get(pred, '?')}, "
              f"Expected={label_names.get(exp, '?')} {status}")
    
    # Calculate accuracy
    accuracy = np.mean(board_labels == np.array(expected)) * 100
    print(f"\n[INFO] Classification accuracy: {accuracy:.1f}%")
    
    # Visualize clusters
    if config.SHOW_DEBUG_WINDOWS:
        raw_labels = classifier.kmeans.labels_
        vis = classifier.get_cluster_visualization(all_features, raw_labels)
        cv2.imshow("K-means Clusters", vis)
        
        print("\nPress any key to close window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n[INFO] K-means classifier test completed")


if __name__ == "__main__":
    test_kmeans_classifier()
