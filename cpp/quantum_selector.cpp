#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <execution>
#include <immintrin.h>  // Pour SIMD AVX2
#include <omp.h>        // Pour OpenMP

namespace py = pybind11;

// Configuration compile-time
constexpr size_t SIMD_WIDTH = 4;  // AVX2 = 4 doubles
constexpr size_t CACHE_LINE = 64;
constexpr double EPSILON = 1e-10;

// Structure optimisée pour l'alignement mémoire
struct alignas(32) AlignedVector {
    std::vector<double> data;
    
    AlignedVector(size_t size) : data(size) {}
    double* get() { return data.data(); }
    const double* get() const { return data.data(); }
    size_t size() const { return data.size(); }
};

// ===== FONCTIONS DE DISTANCE OPTIMISÉES =====

// Distance L1 (Manhattan) avec SIMD AVX2
inline double l1_distance_simd(const double* a, const double* b, size_t n) {
    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;
    
    // Traitement vectoriel par blocs de 4
    for (; i + 3 < n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d diff = _mm256_sub_pd(va, vb);
        
        // Valeur absolue vectorielle
        __m256d mask = _mm256_set1_pd(-0.0);
        diff = _mm256_andnot_pd(mask, diff);
        
        sum = _mm256_add_pd(sum, diff);
    }
    
    // Réduction horizontale
    double result[4];
    _mm256_storeu_pd(result, sum);
    double total = result[0] + result[1] + result[2] + result[3];
    
    // Traitement du reste
    for (; i < n; ++i) {
        total += std::abs(a[i] - b[i]);
    }
    
    return total;
}

// Distance L2 (Euclidienne) avec SIMD AVX2
inline double l2_distance_simd(const double* a, const double* b, size_t n) {
    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;
    
    for (; i + 3 < n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d diff = _mm256_sub_pd(va, vb);
        __m256d squared = _mm256_mul_pd(diff, diff);
        sum = _mm256_add_pd(sum, squared);
    }
    
    double result[4];
    _mm256_storeu_pd(result, sum);
    double total = result[0] + result[1] + result[2] + result[3];
    
    for (; i < n; ++i) {
        double diff = a[i] - b[i];
        total += diff * diff;
    }
    
    return std::sqrt(total);
}

// Distance cosinus optimisée
inline double cosine_distance_simd(const double* a, const double* b, size_t n) {
    __m256d dot_sum = _mm256_setzero_pd();
    __m256d norm_a_sum = _mm256_setzero_pd();
    __m256d norm_b_sum = _mm256_setzero_pd();
    
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        
        dot_sum = _mm256_fmadd_pd(va, vb, dot_sum);  // a*b + sum
        norm_a_sum = _mm256_fmadd_pd(va, va, norm_a_sum);
        norm_b_sum = _mm256_fmadd_pd(vb, vb, norm_b_sum);
    }
    
    // Réduction
    double dot[4], norm_a[4], norm_b[4];
    _mm256_storeu_pd(dot, dot_sum);
    _mm256_storeu_pd(norm_a, norm_a_sum);
    _mm256_storeu_pd(norm_b, norm_b_sum);
    
    double dot_product = dot[0] + dot[1] + dot[2] + dot[3];
    double norm_a_total = norm_a[0] + norm_a[1] + norm_a[2] + norm_a[3];
    double norm_b_total = norm_b[0] + norm_b[1] + norm_b[2] + norm_b[3];
    
    // Reste
    for (; i < n; ++i) {
        dot_product += a[i] * b[i];
        norm_a_total += a[i] * a[i];
        norm_b_total += b[i] * b[i];
    }
    
    double denominator = std::sqrt(norm_a_total * norm_b_total);
    if (denominator < EPSILON) return 1.0;  // Vecteurs orthogonaux
    
    double cosine_sim = dot_product / denominator;
    return 1.0 - std::max(-1.0, std::min(1.0, cosine_sim));
}

// ===== ALGORITHMES DE SCORING AVANCÉS =====

// Scoring quantique avec pondération adaptative
inline double quantum_weighted_score(const double* state, const double* future, 
                                    const double* weights, size_t n) {
    __m256d score_sum = _mm256_setzero_pd();
    
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d vs = _mm256_loadu_pd(&state[i]);
        __m256d vf = _mm256_loadu_pd(&future[i]);
        __m256d vw = _mm256_loadu_pd(&weights[i]);
        
        __m256d diff = _mm256_sub_pd(vf, vs);
        __m256d abs_diff = _mm256_andnot_pd(_mm256_set1_pd(-0.0), diff);
        __m256d weighted = _mm256_mul_pd(abs_diff, vw);
        
        score_sum = _mm256_add_pd(score_sum, weighted);
    }
    
    double result[4];
    _mm256_storeu_pd(result, score_sum);
    double total = result[0] + result[1] + result[2] + result[3];
    
    for (; i < n; ++i) {
        total += std::abs(future[i] - state[i]) * weights[i];
    }
    
    return total;
}

// ===== FONCTIONS PRINCIPALES OPTIMISÉES =====

// Version ultra-rapide avec choix de métrique
std::vector<double> quantum_score_optimized(
    const std::vector<std::vector<double>>& states,
    const std::vector<std::vector<double>>& futures,
    const std::string& metric = "l1",
    int num_threads = -1) {
    
    if (states.empty() || futures.empty()) {
        return std::vector<double>();
    }
    
    const size_t n_futures = futures.size();
    const size_t n_features = states[0].size();
    const double* reference_state = states[0].data();
    
    // Configuration OpenMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    std::vector<double> scores(n_futures);
    
    // Parallélisation OpenMP avec scheduling optimisé
    #pragma omp parallel for schedule(static, 64) num_threads(num_threads > 0 ? num_threads : omp_get_max_threads())
    for (size_t i = 0; i < n_futures; ++i) {
        const double* future_data = futures[i].data();
        
        if (metric == "l1") {
            scores[i] = l1_distance_simd(reference_state, future_data, n_features);
        } else if (metric == "l2") {
            scores[i] = l2_distance_simd(reference_state, future_data, n_features);
        } else if (metric == "cosine") {
            scores[i] = cosine_distance_simd(reference_state, future_data, n_features);
        } else {
            // Fallback vers L1
            scores[i] = l1_distance_simd(reference_state, future_data, n_features);
        }
    }
    
    return scores;
}

// Version avec scoring pondéré et multi-états
std::vector<double> quantum_score_advanced(
    const std::vector<std::vector<double>>& states,
    const std::vector<std::vector<double>>& futures,
    const std::vector<double>& weights,
    const std::vector<double>& state_weights,
    bool normalize = true) {
    
    if (states.empty() || futures.empty()) {
        return std::vector<double>();
    }
    
    const size_t n_futures = futures.size();
    const size_t n_features = states[0].size();
    const size_t n_states = states.size();
    
    // Vérification des dimensions
    if (!weights.empty() && weights.size() != n_features) {
        throw std::invalid_argument("Weights size mismatch");
    }
    if (!state_weights.empty() && state_weights.size() != n_states) {
        throw std::invalid_argument("State weights size mismatch");
    }
    
    std::vector<double> scores(n_futures, 0.0);
    const double* weight_data = weights.empty() ? nullptr : weights.data();
    
    #pragma omp parallel for schedule(dynamic, 32)
    for (size_t i = 0; i < n_futures; ++i) {
        const double* future_data = futures[i].data();
        double total_score = 0.0;
        double total_weight = 0.0;
        
        // Scoring contre tous les états avec pondération
        for (size_t s = 0; s < n_states; ++s) {
            const double* state_data = states[s].data();
            double state_weight = state_weights.empty() ? 1.0 : state_weights[s];
            
            double score;
            if (weight_data) {
                score = quantum_weighted_score(state_data, future_data, weight_data, n_features);
            } else {
                score = l1_distance_simd(state_data, future_data, n_features);
            }
            
            total_score += score * state_weight;
            total_weight += state_weight;
        }
        
        scores[i] = total_weight > 0 ? total_score / total_weight : 0.0;
    }
    
    // Normalisation optionnelle
    if (normalize && !scores.empty()) {
        auto [min_it, max_it] = std::minmax_element(scores.begin(), scores.end());
        double range = *max_it - *min_it;
        if (range > EPSILON) {
            std::transform(std::execution::par_unseq, scores.begin(), scores.end(), scores.begin(),
                          [min_val = *min_it, range](double score) {
                              return (score - min_val) / range;
                          });
        }
    }
    
    return scores;
}

// Version avec support NumPy natif (plus rapide depuis Python)
py::array_t<double> quantum_score_numpy(py::array_t<double> states_np,
                                       py::array_t<double> futures_np,
                                       const std::string& metric = "l1") {
    
    auto states_buf = states_np.request();
    auto futures_buf = futures_np.request();
    
    if (states_buf.ndim != 2 || futures_buf.ndim != 2) {
        throw std::runtime_error("Input arrays must be 2-dimensional");
    }
    
    const size_t n_states = states_buf.shape[0];
    const size_t n_features = states_buf.shape[1];
    const size_t n_futures = futures_buf.shape[0];
    
    if (futures_buf.shape[1] != n_features) {
        throw std::runtime_error("Feature dimension mismatch");
    }
    
    auto result = py::array_t<double>(n_futures);
    auto result_buf = result.request();
    
    const double* states_ptr = static_cast<double*>(states_buf.ptr);
    const double* futures_ptr = static_cast<double*>(futures_buf.ptr);
    double* scores_ptr = static_cast<double*>(result_buf.ptr);
    
    const double* reference_state = states_ptr;  // Premier état comme référence
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_futures; ++i) {
        const double* future_data = futures_ptr + i * n_features;
        
        if (metric == "l1") {
            scores_ptr[i] = l1_distance_simd(reference_state, future_data, n_features);
        } else if (metric == "l2") {
            scores_ptr[i] = l2_distance_simd(reference_state, future_data, n_features);
        } else if (metric == "cosine") {
            scores_ptr[i] = cosine_distance_simd(reference_state, future_data, n_features);
        } else {
            scores_ptr[i] = l1_distance_simd(reference_state, future_data, n_features);
        }
    }
    
    return result;
}

// ===== BINDING PYBIND11 =====

PYBIND11_MODULE(quantum_selector, m) {
    m.doc() = "Module C++ ultra-optimisé pour le scoring quantique (SIMD + OpenMP + pybind11)";
    
    // Version de base (rétrocompatible)
    m.def("quantum_score", &quantum_score_optimized, 
          "Scoring quantique optimisé avec SIMD et OpenMP",
          py::arg("states"), py::arg("futures"), 
          py::arg("metric") = "l1", py::arg("num_threads") = -1);
    
    // Version avancée
    m.def("quantum_score_advanced", &quantum_score_advanced,
          "Scoring quantique avancé avec pondération multi-états",
          py::arg("states"), py::arg("futures"),
          py::arg("weights") = std::vector<double>(),
          py::arg("state_weights") = std::vector<double>(),
          py::arg("normalize") = true);
    
    // Version NumPy native
    m.def("quantum_score_numpy", &quantum_score_numpy,
          "Scoring quantique avec support NumPy natif",
          py::arg("states"), py::arg("futures"), py::arg("metric") = "l1");
    
    // Métriques disponibles
    m.attr("SUPPORTED_METRICS") = py::make_tuple("l1", "l2", "cosine");
}