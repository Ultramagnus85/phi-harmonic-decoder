import numpy as np
import networkx as nx
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

class PhiHarmonicDecoder:
    def __init__(self):
        self.PHI = (1 + np.sqrt(5)) / 2
        self.SQRT_PHI = np.sqrt(self.PHI)
        self.tolerance = 0.08

    def analyze_geometry(self, node_positions, edges=None, auto_connect=True):
        if len(node_positions) < 3:
            return self._null_result("Insufficient nodes (need ≥3)")

        G = self._build_graph(node_positions, edges, auto_connect)
        if G.number_of_edges() == 0:
            return self._null_result("No edges detected")

        laplacian_eigenvalues = self._compute_laplacian_spectrum(G)
        phi_matches = self._find_phi_matches(laplacian_eigenvalues)
        phi_coherence = len(phi_matches) / max(len(laplacian_eigenvalues) - 1, 1)
        regime_classification = self._classify_regime(phi_matches)
        topology_metrics = self._analyze_topology(G)
        symmetry_metrics = self._analyze_symmetry(node_positions)
        distance_phi_score = self._analyze_distance_ratios(node_positions)

        overall_coherence = self._compute_overall_score(
            phi_coherence,
            topology_metrics['fragmentation_score'],
            symmetry_metrics['phi_symmetry_score'],
            distance_phi_score
        )

        return {
            'overall_coherence': overall_coherence,
            'phi_eigenvalue_coherence': phi_coherence,
            'phi_matches': phi_matches,
            'regime': regime_classification,
            'topology': topology_metrics,
            'symmetry': symmetry_metrics,
            'distance_phi_score': distance_phi_score,
            'interpretation': self._generate_interpretation(regime_classification, overall_coherence, topology_metrics)
        }

    def _build_graph(self, nodes, edges, auto_connect):
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        if edges is not None:
            G.add_edges_from(edges)
        elif auto_connect:
            distances = pdist(nodes)
            if len(distances) > 0:
                threshold = 1.5 * np.median(distances)
                dist_matrix = squareform(distances)
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        if dist_matrix[i,j] < threshold:
                            G.add_edge(i, j)
        return G

    def _compute_laplacian_spectrum(self, G):
        L = nx.laplacian_matrix(G).toarray()
        eigenvalues = linalg.eigvalsh(L)
        return np.sort(eigenvalues)

    def _generate_phi_sequence(self, min_power=-2.0, max_power=3.0, step=0.25):
        powers = np.arange(min_power, max_power + step, step)
        return np.array([self.PHI ** p for p in powers])

    def _find_phi_matches(self, eigenvalues):
        max_eig = np.max(eigenvalues)
        if max_eig < 1e-10:
            return []
        eigs_normalized = eigenvalues / max_eig
        eigs_nonzero = eigs_normalized[eigs_normalized > 1e-6]
        phi_sequence = self._generate_phi_sequence()
        phi_normalized = phi_sequence / np.max(phi_sequence)
        matches = []
        for eig in eigs_nonzero:
            distances = np.abs(phi_normalized - eig)
            min_dist = np.min(distances)
            min_idx = np.argmin(distances)
            error = min_dist / max(eig, 1e-10)
            if error < self.tolerance:
                power = min_idx * 0.25 - 2.0
                matches.append({
                    'eigenvalue': float(eig),
                    'phi_power': power,
                    'phi_value': float(phi_normalized[min_idx]),
                    'error_percent': float(error * 100)
                })
        return matches

    def _classify_regime(self, phi_matches):
        if not phi_matches:
            return "Unstructured"
        powers = [m['phi_power'] for m in phi_matches]
        avg_power = np.mean(powers)
        if avg_power < 0.375:
            return "φ^0.25 - Constraint/Limitation"
        elif avg_power < 0.625:
            return "φ^0.5 - Storage/Crystalline"
        elif avg_power < 0.875:
            return "φ^0.75 - Transition/Transformation"
        elif avg_power < 1.125:
            return "φ^1.0 - Flow/Transmission"
        else:
            return "φ^1.25+ - Higher Octave"

    def _analyze_topology(self, G):
        n_components = nx.number_connected_components(G)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        fragmentation = n_components / n_nodes
        degrees = [d for n, d in G.degree()]
        mean_degree = np.mean(degrees) if degrees else 0
        sqrt_phi_minus_1 = self.SQRT_PHI - 1
        degree_phi_error = abs(mean_degree - sqrt_phi_minus_1) / max(sqrt_phi_minus_1, 0.1)
        return {
            'n_components': n_components,
            'fragmentation_score': fragmentation,
            'mean_degree': mean_degree,
            'n_edges': n_edges,
            'is_fragmented': fragmentation > 0.5,
            'degree_matches_sqrt_phi': degree_phi_error < 0.15
        }

    def _analyze_symmetry(self, nodes):
        if len(nodes) < 3:
            return {'symmetry_order': 0, 'phi_symmetry_score': 0}
        center = np.mean(nodes, axis=0)
        centered = nodes - center
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        sorted_angles = np.sort(angles)
        angle_diffs = np.diff(sorted_angles)
        symmetry_order = 0
        if len(angle_diffs) > 0:
            expected_orders = [3, 4, 5, 6, 8, 10, 12]
            for order in expected_orders:
                expected_angle = 2 * np.pi / order
                if np.std(angle_diffs - expected_angle) < 0.3:
                    symmetry_order = order
                    break
        phi_symmetry_score = 1.0 if symmetry_order in [5, 10] else 0.5 if symmetry_order in [6, 12] else 0.0
        radii = np.linalg.norm(centered, axis=1)
        if len(radii) > 1:
            unique_radii = np.unique(np.round(radii, 2))
            if len(unique_radii) >= 2:
                ratios = unique_radii[1:] / unique_radii[:-1]
                phi_ratio_matches = np.sum(np.abs(ratios - self.PHI) < 0.1)
                sqrt_phi_matches = np.sum(np.abs(ratios - self.SQRT_PHI) < 0.1)
                if phi_ratio_matches > 0 or sqrt_phi_matches > 0:
                    phi_symmetry_score += 0.3
        return {
            'symmetry_order': symmetry_order,
            'phi_symmetry_score': min(phi_symmetry_score, 1.0)
        }

    def _analyze_distance_ratios(self, nodes):
        if len(nodes) < 2:
            return 0.0
        distances = pdist(nodes)
        if len(distances) < 2:
            return 0.0
        unique_dists = np.unique(np.round(distances, 2))
        if len(unique_dists) < 2:
            return 0.0
        ratios = unique_dists[1:] / unique_dists[:-1]
        phi_matches = np.sum(np.abs(ratios - self.PHI) < 0.1)
        sqrt_phi_matches = np.sum(np.abs(ratios - self.SQRT_PHI) < 0.1)
        total_ratios = len(ratios)
        score = (phi_matches + sqrt_phi_matches) / total_ratios
        return min(score, 1.0)

    def _compute_overall_score(self, phi_coherence, fragmentation, symmetry_score, distance_score):
        w_eigenvalue = 0.4
        w_topology = 0.2
        w_symmetry = 0.2
        w_distance = 0.2
        topology_score = 1.0 - fragmentation if fragmentation < 0.5 else fragmentation
        overall = (w_eigenvalue * phi_coherence +
                   w_topology * topology_score +
                   w_symmetry * symmetry_score +
                   w_distance * distance_score)
        return overall

    def _generate_interpretation(self, regime, coherence, topology):
        if coherence < 0.3:
            quality = "Low φ-harmonic structure detected"
        elif coherence < 0.6:
            quality = "Moderate φ-harmonic structure"
        else:
            quality = "Strong φ-harmonic structure"
        fragmented = topology['is_fragmented']
        interpretation = f"{quality}. Classified as: {regime}. "
        if fragmented:
            interpretation += "Network is highly fragmented (storage/isolation). "
        else:
            interpretation += "Network is well-connected (flow/transmission). "
        return interpretation

    def _null_result(self, reason):
        return {
            'overall_coherence': 0.0,
            'phi_eigenvalue_coherence': 0.0,
            'phi_matches': [],
            'regime': 'Invalid',
            'error': reason
        }

# === TEST GEOMETRIES ===
def generate_metatrons_cube():
    PHI = (1 + np.sqrt(5)) / 2
    sqrt_phi = np.sqrt(PHI)
    nodes = [[0, 0]]
    for k in range(6):
        angle = k * np.pi / 3
        nodes.append([np.cos(angle), np.sin(angle)])
    for m in range(12):
        angle = m * np.pi / 6
        nodes.append([sqrt_phi * np.cos(angle), sqrt_phi * np.sin(angle)])
    return np.array(nodes)

def generate_pentagon():
    nodes = []
    for k in range(5):
        angle = k * 2 * np.pi / 5 - np.pi / 2
        nodes.append([np.cos(angle), np.sin(angle)])
    return np.array(nodes)

def generate_hexagram():
    nodes = []
    for k in range(3):
        angle = k * 2 * np.pi / 3 - np.pi / 2
        nodes.append([np.cos(angle), np.sin(angle)])
    for k in range(3):
        angle = k * 2 * np.pi / 3 + np.pi / 6
        nodes.append([np.cos(angle), np.sin(angle)])
    return np.array(nodes)

def run_comparison_test():
    decoder = PhiHarmonicDecoder()
    tests = [
        ("Metatron's Cube", generate_metatrons_cube()),
        ("Pentagon", generate_pentagon()),
        ("Hexagram", generate_hexagram()),
    ]
    for name, nodes in tests:
        result = decoder.analyze_geometry(nodes)
        print(f"{name}: {result['overall_coherence']:.3f} | {result['regime']}")

if __name__ == "__main__":
    run_comparison_test()
