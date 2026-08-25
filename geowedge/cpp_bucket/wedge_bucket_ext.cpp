// wedge_bucket_ext.cpp — C++17 implementation of wedge bucket search.
//
// P1-P5 filter + log-bucket search fused into one C++ entry point.
//
// Design differences from the numba version:
//   - std::vector<double> for state arrays (no numpy overhead)
//   - std::unordered_map<int64_t, ...> for bucket group-by (O(n) avg
//     vs the numba sort-based O(n log n) group-by)
//   - std::partial_sort for the max_states cap (O(n log k) vs
//     O(n log n) full sort)
//   - Native double math, -O3 -march=native compile flags
//
// Recall / algorithmic behavior: bit-exact match to numba v3 modulo
// FP reordering that -ffast-math permits.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

// =========================================================================
// P1-P5 filter
// =========================================================================
static std::tuple<int, std::vector<double>, std::vector<double>, double, double>
check_rules_cpp(std::vector<double> in_arr, std::vector<double> out_arr,
                bool anchor_side_is_in, double anchor_val,
                double theta, double eps,
                double sum_in, double sum_out) {
    const double one_minus_eps = 1.0 - eps;
    const double one_plus_eps  = 1.0 + eps;
    const double theta_lower   = one_minus_eps * theta;

    int n_in  = static_cast<int>(in_arr.size());
    int n_out = static_cast<int>(out_arr.size());

    for (int iter = 0; iter < 20; ++iter) {
        // ---- rule1_4 O(1) checks ----
        if (anchor_side_is_in) {
            if (n_out == 0) return {0, in_arr, out_arr, sum_in, sum_out};
            if (sum_in + anchor_val < theta)
                return {0, in_arr, out_arr, sum_in, sum_out};
            if (sum_out < theta_lower)
                return {0, in_arr, out_arr, sum_in, sum_out};
            if (anchor_val * one_minus_eps > sum_out)
                return {0, in_arr, out_arr, sum_in, sum_out};
            double min_out = out_arr[0];
            for (int i = 1; i < n_out; ++i)
                if (out_arr[i] < min_out) min_out = out_arr[i];
            if (min_out > one_plus_eps * (sum_in + anchor_val))
                return {0, in_arr, out_arr, sum_in, sum_out};
        } else {
            if (n_in == 0) return {0, in_arr, out_arr, sum_in, sum_out};
            if (sum_out + anchor_val < theta_lower)
                return {0, in_arr, out_arr, sum_in, sum_out};
            if (sum_in < theta)
                return {0, in_arr, out_arr, sum_in, sum_out};
            double min_in = in_arr[0];
            for (int i = 1; i < n_in; ++i)
                if (in_arr[i] < min_in) min_in = in_arr[i];
            if (min_in * one_minus_eps > sum_out + anchor_val)
                return {0, in_arr, out_arr, sum_in, sum_out};
            if (anchor_val > one_plus_eps * sum_in)
                return {0, in_arr, out_arr, sum_in, sum_out};
        }

        // ---- P4/P5 pruning ----
        double p4_rhs, p5_rhs;
        if (anchor_side_is_in) {
            p4_rhs = sum_out;
            p5_rhs = one_plus_eps * (sum_in + anchor_val);
        } else {
            p4_rhs = sum_out + anchor_val;
            p5_rhs = one_plus_eps * sum_in;
        }

        std::vector<double> new_in, new_out;
        new_in.reserve(n_in);
        new_out.reserve(n_out);
        double new_sum_in = 0.0, new_sum_out = 0.0;

        for (int i = 0; i < n_in; ++i) {
            if (one_minus_eps * in_arr[i] <= p4_rhs) {
                new_in.push_back(in_arr[i]);
                new_sum_in += in_arr[i];
            }
        }
        if (anchor_side_is_in) {
            for (int i = 0; i < n_out; ++i) {
                if (out_arr[i] <= p5_rhs) {
                    new_out.push_back(out_arr[i]);
                    new_sum_out += out_arr[i];
                }
            }
        } else {
            for (int i = 0; i < n_out; ++i) {
                if (out_arr[i] + anchor_val <= p5_rhs) {
                    new_out.push_back(out_arr[i]);
                    new_sum_out += out_arr[i];
                }
            }
        }

        if (static_cast<int>(new_in.size()) == n_in &&
            static_cast<int>(new_out.size()) == n_out) {
            return {-1, in_arr, out_arr, sum_in, sum_out};
        }

        in_arr  = std::move(new_in);
        out_arr = std::move(new_out);
        n_in    = static_cast<int>(in_arr.size());
        n_out   = static_cast<int>(out_arr.size());
        sum_in  = new_sum_in;
        sum_out = new_sum_out;
    }

    return {-1, in_arr, out_arr, sum_in, sum_out};
}


// =========================================================================
// Bucket compress + alive filter (per-iteration workhorse)
// =========================================================================
enum class CompressionMode {
    LogMD = 0,          // log buckets in (M^in, |D|)
    UniformMinMout = 1, // uniform buckets in (M^in, M^out)
    UniformMinD = 2,    // uniform buckets in (M^in, |D|)
};

static CompressionMode parse_compression_mode(const std::string& mode) {
    if (mode == "log" || mode == "log_d" || mode == "log_md") {
        return CompressionMode::LogMD;
    }
    if (mode == "uniform_mout" || mode == "uniform_min_mout" ||
        mode == "min_mout") {
        return CompressionMode::UniformMinMout;
    }
    if (mode == "uniform_d" || mode == "uniform_min_d" ||
        mode == "min_d") {
        return CompressionMode::UniformMinD;
    }
    throw std::invalid_argument("unknown compression_mode: " + mode);
}

struct Best {
    double phi;
    double sa;
    std::size_t idx;
};

static void bucket_compress_and_alive_cpp(
        const std::vector<double>& sas_in,
        const std::vector<double>& sbs_in,
        double theta, double eps,
        double inv_log_sa, double inv_log_d,
        double delta_sa, double delta_d,
        CompressionMode compression_mode,
        double rem_in_next, double rem_out_next,
        int max_states,
        std::vector<double>& out_sas,
        std::vector<double>& out_sbs) {
    const std::size_t n = sas_in.size();
    out_sas.clear();
    out_sbs.clear();
    if (n == 0) return;

    const double one_minus_eps = 1.0 - eps;
    const double one_plus_eps  = 1.0 + eps;
    const double uniform_sa_width = std::max(delta_sa * theta, 1e-9);
    const double uniform_d_width  = std::max(delta_d  * theta, 1e-9);

    // sign encoded as top bit; sa_bucket and d_bucket in 21-bit slots each
    constexpr int64_t OFFSET = 1LL << 20;

    // O(n) average bucket group-by via hash map.
    std::unordered_map<int64_t, Best> buckets;
    buckets.reserve(n * 2);

    for (std::size_t i = 0; i < n; ++i) {
        const double sa = sas_in[i];
        const double sb = sbs_in[i];
        const double gap = sa - sb;
        int64_t sign;
        double d;
        if (gap >= 0.0) { d = gap;  sign = 1; }
        else            { d = -gap; sign = 0; }

        int64_t sa_bucket = 0;
        int64_t d_bucket = 0;
        if (compression_mode == CompressionMode::LogMD) {
            sa_bucket = (sa > 0.0)
                ? static_cast<int64_t>(std::floor(std::log(sa) * inv_log_sa))
                : 0;
            d_bucket = (d > 0.0)
                ? static_cast<int64_t>(std::floor(std::log(d) * inv_log_d))
                : 0;
        } else if (compression_mode == CompressionMode::UniformMinMout) {
            sa_bucket = static_cast<int64_t>(std::floor(sa / uniform_sa_width));
            d_bucket = static_cast<int64_t>(std::floor(sb / uniform_sa_width));
        } else {
            sa_bucket = static_cast<int64_t>(std::floor(sa / uniform_sa_width));
            d_bucket = static_cast<int64_t>(std::floor(d / uniform_d_width));
        }

        const int64_t key = (sign << 42)
                          | ((sa_bucket + OFFSET) << 21)
                          | (d_bucket + OFFSET);

        // Compute phi (L¹ distance to wedge)
        double phi = (sa < theta) ? (theta - sa) : 0.0;
        const double lower = one_minus_eps * sa;
        if (sb < lower) {
            phi += lower - sb;
        } else {
            const double upper = one_plus_eps * sa;
            if (sb > upper) phi += sb - upper;
        }

        auto it = buckets.find(key);
        if (it == buckets.end()) {
            buckets.emplace(key, Best{phi, sa, i});
        } else {
            if (phi < it->second.phi ||
                (phi == it->second.phi && sa > it->second.sa)) {
                it->second = Best{phi, sa, i};
            }
        }
    }

    // Extract kept indices
    std::vector<std::size_t> kept_idx;
    kept_idx.reserve(buckets.size());
    std::vector<double> kept_phi;
    kept_phi.reserve(buckets.size());
    for (const auto& [key, best] : buckets) {
        kept_idx.push_back(best.idx);
        kept_phi.push_back(best.phi);
    }

    // Max states cap via partial_sort on phi
    if (static_cast<int>(kept_idx.size()) > max_states) {
        std::vector<std::size_t> order(kept_idx.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::partial_sort(order.begin(),
                          order.begin() + max_states,
                          order.end(),
                          [&](std::size_t a, std::size_t b) {
                              return kept_phi[a] < kept_phi[b];
                          });
        std::vector<std::size_t> trimmed;
        trimmed.reserve(max_states);
        for (int i = 0; i < max_states; ++i) trimmed.push_back(kept_idx[order[i]]);
        kept_idx = std::move(trimmed);
    }

    // Alive filter + write out
    out_sas.reserve(kept_idx.size());
    out_sbs.reserve(kept_idx.size());
    for (const std::size_t idx : kept_idx) {
        const double sa = sas_in[idx];
        const double sb = sbs_in[idx];
        if (sa + rem_in_next < theta)                       continue;
        if (sb + rem_out_next < sa * one_minus_eps)         continue;
        if (sb > one_plus_eps * (sa + rem_in_next))         continue;
        out_sas.push_back(sa);
        out_sbs.push_back(sb);
    }
}


// =========================================================================
// Main bucket search core
// =========================================================================
struct Result {
    bool found;
    double sa;
    double sb;
};

static Result core_bucket_search_cpp(
        double initial_sa, double initial_sb,
        const std::vector<double>& cand_amts,
        const std::vector<int8_t>&  cand_is_in,
        double theta, double eps,
        double delta_sa, double delta_d,
        int max_states,
        CompressionMode compression_mode = CompressionMode::LogMD) {
    const int n = static_cast<int>(cand_amts.size());
    const double one_minus_eps = 1.0 - eps;
    const double eps_relaxed   = eps * (1.0 + delta_d);
    const double inv_log_sa    = 1.0 / std::log(1.0 + delta_sa);
    const double inv_log_d     = 1.0 / std::log(1.0 + delta_d);

    // Suffix reachability sums
    std::vector<double> rem_in(n + 1, 0.0), rem_out(n + 1, 0.0);
    for (int k = n - 1; k >= 0; --k) {
        if (cand_is_in[k]) {
            rem_in[k]  = rem_in[k+1] + cand_amts[k];
            rem_out[k] = rem_out[k+1];
        } else {
            rem_in[k]  = rem_in[k+1];
            rem_out[k] = rem_out[k+1] + cand_amts[k];
        }
    }

    // Anchor-only witness
    if (initial_sa >= theta) {
        const double g0 = std::abs(initial_sa - initial_sb);
        if (g0 <= eps_relaxed * initial_sa) return {true, initial_sa, initial_sb};
    }

    // Initial alive check
    if (initial_sa + rem_in[0] < theta) return {false, 0.0, 0.0};
    if (initial_sb + rem_out[0] < initial_sa * one_minus_eps)
        return {false, 0.0, 0.0};

    std::vector<double> sas{initial_sa};
    std::vector<double> sbs{initial_sb};

    // Reusable buffers for expand
    std::vector<double> new_sas, new_sbs;
    std::vector<double> compressed_sas, compressed_sbs;

    for (int k = 0; k < n; ++k) {
        const double amt = cand_amts[k];
        const bool  is_in = cand_is_in[k] != 0;
        const int m = static_cast<int>(sas.size());

        // Expand (skip + add)
        new_sas.resize(2 * m);
        new_sbs.resize(2 * m);
        if (is_in) {
            for (int i = 0; i < m; ++i) {
                new_sas[i]     = sas[i];
                new_sas[i + m] = sas[i] + amt;
                new_sbs[i]     = sbs[i];
                new_sbs[i + m] = sbs[i];
            }
        } else {
            for (int i = 0; i < m; ++i) {
                new_sas[i]     = sas[i];
                new_sas[i + m] = sas[i];
                new_sbs[i]     = sbs[i];
                new_sbs[i + m] = sbs[i] + amt;
            }
        }

        // Compress + alive
        bucket_compress_and_alive_cpp(
            new_sas, new_sbs, theta, eps,
            inv_log_sa, inv_log_d,
            delta_sa, delta_d,
            compression_mode,
            rem_in[k+1], rem_out[k+1],
            max_states,
            compressed_sas, compressed_sbs
        );

        if (compressed_sas.empty()) return {false, 0.0, 0.0};

        // exact_check_relaxed
        for (std::size_t i = 0; i < compressed_sas.size(); ++i) {
            const double sa = compressed_sas[i];
            const double sb = compressed_sbs[i];
            if (sa >= theta) {
                const double g = std::abs(sa - sb);
                if (g <= eps_relaxed * sa) return {true, sa, sb};
            }
        }

        sas.swap(compressed_sas);
        sbs.swap(compressed_sbs);
    }

    return {false, 0.0, 0.0};
}


// =========================================================================
// Adaptive-max greedy phase.
// =========================================================================
//
// Mirrors the adaptive-max greedy logic used by the cascade.
//
// Rule at each step:
//   * If SA <= SB: grow SA by adding an IN candidate.
//   * If SA >  SB: grow SB by adding an OUT candidate.
//   * Among candidates that don't overshoot the wedge upper bound,
//     pick the LARGEST. If none fit, take the SMALLEST (least overshoot).
//   * Check wedge after each add.
//
// Returns (found, sa, sb). Uses swap-and-pop_back for O(1) removal.
// =========================================================================
static Result adaptive_max_greedy_cpp(
        std::vector<double>& in_arr,
        std::vector<double>& out_arr,
        bool anchor_side_is_in,
        double anchor_val,
        double theta, double eps, double delta_d) {
    const double eps_relaxed  = eps * (1.0 + delta_d);
    const double one_minus_eps = 1.0 - eps;
    const double one_plus_eps  = 1.0 + eps;

    double sa, sb;
    if (anchor_side_is_in) { sa = anchor_val; sb = 0.0; }
    else                   { sa = 0.0;        sb = anchor_val; }

    // Initial wedge check (anchor alone)
    if (sa >= theta && std::abs(sb - sa) <= eps_relaxed * sa) {
        return {true, sa, sb};
    }

    double sum_remaining_in = 0.0;
    for (double v : in_arr) sum_remaining_in += v;

    while (!in_arr.empty() || !out_arr.empty()) {
        const double max_possible_sa = sa + sum_remaining_in;

        if (sa <= sb) {
            // Grow SA (add IN)
            if (in_arr.empty()) break;

            double max_add;
            if (sb > 0.0) max_add = sb / one_minus_eps - sa;
            else          max_add = std::numeric_limits<double>::max();

            // Find largest eligible (<= max_add)
            int best_idx = -1;
            double best_val = -1.0;
            for (int i = 0; i < static_cast<int>(in_arr.size()); ++i) {
                const double v = in_arr[i];
                if (v <= max_add && v > best_val) {
                    best_val = v;
                    best_idx = i;
                }
            }
            if (best_idx < 0) {
                // No eligible; take smallest
                best_idx = 0;
                double smallest = in_arr[0];
                for (int i = 1; i < static_cast<int>(in_arr.size()); ++i) {
                    if (in_arr[i] < smallest) {
                        smallest = in_arr[i];
                        best_idx = i;
                    }
                }
            }
            const double choice = in_arr[best_idx];
            sa += choice;
            sum_remaining_in -= choice;
            // O(1) removal: swap-with-back + pop
            in_arr[best_idx] = in_arr.back();
            in_arr.pop_back();
        } else {
            // Grow SB (add OUT)
            if (out_arr.empty()) break;

            const double max_add = max_possible_sa * one_plus_eps - sb;

            int best_idx = -1;
            double best_val = -1.0;
            for (int i = 0; i < static_cast<int>(out_arr.size()); ++i) {
                const double v = out_arr[i];
                if (v <= max_add && v > best_val) {
                    best_val = v;
                    best_idx = i;
                }
            }
            if (best_idx < 0) {
                best_idx = 0;
                double smallest = out_arr[0];
                for (int i = 1; i < static_cast<int>(out_arr.size()); ++i) {
                    if (out_arr[i] < smallest) {
                        smallest = out_arr[i];
                        best_idx = i;
                    }
                }
            }
            const double choice = out_arr[best_idx];
            sb += choice;
            out_arr[best_idx] = out_arr.back();
            out_arr.pop_back();
        }

        // Wedge check after each add
        if (sa >= theta && std::abs(sb - sa) <= eps_relaxed * sa) {
            return {true, sa, sb};
        }
    }

    return {false, 0.0, 0.0};
}


// =========================================================================
// Greedy-only public entry (filter + adaptive-max greedy, no bucket)
// =========================================================================
static py::object frontier_search_greedyonly_cpp(
        py::array_t<double, py::array::c_style | py::array::forcecast> in_amts,
        py::array_t<double, py::array::c_style | py::array::forcecast> out_amts,
        double anchor_val,
        std::string anchor_side,
        double theta, double eps,
        double delta_d) {
    const int n_in  = static_cast<int>(in_amts.size());
    const int n_out = static_cast<int>(out_amts.size());

    std::vector<double> in_vec(in_amts.data(),  in_amts.data() + n_in);
    std::vector<double> out_vec(out_amts.data(), out_amts.data() + n_out);

    double sum_in = 0.0, sum_out = 0.0;
    for (double v : in_vec)  sum_in  += v;
    for (double v : out_vec) sum_out += v;

    const bool anchor_side_is_in = (anchor_side == "in");

    // Filter (P1-P5) — reuse check_rules_cpp
    auto [flag, in_pruned, out_pruned, _sum_in_p, _sum_out_p] = check_rules_cpp(
        std::move(in_vec), std::move(out_vec),
        anchor_side_is_in, anchor_val, theta, eps, sum_in, sum_out
    );
    (void)_sum_in_p; (void)_sum_out_p;

    if (flag == 0) return py::none();

    // Adaptive-max greedy (no bucket fallback)
    Result r = adaptive_max_greedy_cpp(
        in_pruned, out_pruned,
        anchor_side_is_in, anchor_val,
        theta, eps, delta_d
    );

    if (!r.found) return py::none();
    return py::make_tuple(r.sa, r.sb);
}


// =========================================================================
// Public entry point (called from Python)
// =========================================================================
static py::object frontier_search_full_cpp(
        py::array_t<double, py::array::c_style | py::array::forcecast> in_amts,
        py::array_t<double, py::array::c_style | py::array::forcecast> out_amts,
        double anchor_val,
        std::string anchor_side,
        double theta, double eps,
        double delta_sa,
        double delta_d,
        int max_states) {
    const int n_in  = static_cast<int>(in_amts.size());
    const int n_out = static_cast<int>(out_amts.size());
    const int n = n_in + n_out;

    // Adaptive parameter scaling (matches all other variants)
    if (n > 3000) {
        delta_sa = std::max(delta_sa, 0.35);
        delta_d  = std::max(delta_d,  0.35);
        max_states = std::min(max_states, 300);
    } else if (n > 1500) {
        delta_sa = std::max(delta_sa, 0.25);
        delta_d  = std::max(delta_d,  0.25);
        max_states = std::min(max_states, 600);
    } else if (n > 500) {
        delta_sa = std::max(delta_sa, 0.18);
        delta_d  = std::max(delta_d,  0.18);
        max_states = std::min(max_states, 1200);
    } else if (n > 100) {
        delta_sa = std::max(delta_sa, 0.13);
        delta_d  = std::max(delta_d,  0.13);
        max_states = std::min(max_states, 2500);
    }

    // Copy numpy arrays into std::vector
    std::vector<double> in_vec(in_amts.data(),  in_amts.data() + n_in);
    std::vector<double> out_vec(out_amts.data(), out_amts.data() + n_out);

    // Compute sums
    double sum_in = 0.0, sum_out = 0.0;
    for (double v : in_vec)  sum_in  += v;
    for (double v : out_vec) sum_out += v;

    const bool anchor_side_is_in = (anchor_side == "in");

    // Filter (P1-P5)
    auto [flag, in_pruned, out_pruned, _sum_in_p, _sum_out_p] = check_rules_cpp(
        std::move(in_vec), std::move(out_vec),
        anchor_side_is_in, anchor_val, theta, eps, sum_in, sum_out
    );
    (void)_sum_in_p; (void)_sum_out_p;

    if (flag == 0) return py::none();

    // Initial state
    double initial_sa, initial_sb;
    if (anchor_side_is_in) { initial_sa = anchor_val; initial_sb = 0.0; }
    else                   { initial_sa = 0.0;        initial_sb = anchor_val; }

    // Anchor-only witness
    if (initial_sa >= theta) {
        const double g0 = std::abs(initial_sa - initial_sb);
        if (g0 <= eps * (1.0 + delta_d) * initial_sa) {
            return py::make_tuple(initial_sa, initial_sb);
        }
    }

    const int n_in_p  = static_cast<int>(in_pruned.size());
    const int n_out_p = static_cast<int>(out_pruned.size());
    const int n_p = n_in_p + n_out_p;
    if (n_p == 0) return py::none();

    // Combine + sort by amount DESCENDING
    std::vector<std::pair<double, int8_t>> combined;
    combined.reserve(n_p);
    for (double a : in_pruned)  combined.emplace_back(a, static_cast<int8_t>(1));
    for (double a : out_pruned) combined.emplace_back(a, static_cast<int8_t>(0));
    std::sort(combined.begin(), combined.end(),
              [](const auto& x, const auto& y) { return x.first > y.first; });

    std::vector<double> cand_amts(n_p);
    std::vector<int8_t> cand_is_in(n_p);
    for (int i = 0; i < n_p; ++i) {
        cand_amts[i]  = combined[i].first;
        cand_is_in[i] = combined[i].second;
    }

    // Bucket search
    Result result = core_bucket_search_cpp(
        initial_sa, initial_sb,
        cand_amts, cand_is_in,
        theta, eps, delta_sa, delta_d, max_states
    );

    if (!result.found) return py::none();
    return py::make_tuple(result.sa, result.sb);
}


// =========================================================================
// CASCADE public entry point.
// =========================================================================
//
// Pipeline:
//   1. Filter (P1-P5)         — shared prep work, prunes bad candidates
//   2. Adaptive-max greedy    — Phase 1, resolves ~90% of anchors quickly
//   3. If greedy fails, bucket search — Phase 2 safety net for edge cases
//
// Why filter FIRST and share between greedy/bucket:
//   * P4/P5 pruning significantly shrinks bucket's state-space work
//   * Greedy on pruned lists is slightly faster (fewer doomed candidates)
//   * check_rules_cpp is cheap in C++ (~3-5 μs); running once and sharing
//     the pruned lists is strictly better than any split design
//   * Matches the cascade semantics exactly.
//
// =========================================================================
static py::object frontier_search_cascade_impl(
        py::array_t<double, py::array::c_style | py::array::forcecast> in_amts,
        py::array_t<double, py::array::c_style | py::array::forcecast> out_amts,
        double anchor_val,
        std::string anchor_side,
        double theta, double eps,
        double delta_sa,
        double delta_d,
        int max_states,
        CompressionMode compression_mode) {
    const int n_in  = static_cast<int>(in_amts.size());
    const int n_out = static_cast<int>(out_amts.size());

    // ---- Copy numpy arrays and compute sums ----
    std::vector<double> in_vec(in_amts.data(),  in_amts.data() + n_in);
    std::vector<double> out_vec(out_amts.data(), out_amts.data() + n_out);

    double sum_in = 0.0, sum_out = 0.0;
    for (double v : in_vec)  sum_in  += v;
    for (double v : out_vec) sum_out += v;

    const bool anchor_side_is_in = (anchor_side == "in");

    // Preserve the ORIGINAL caller-supplied delta_d so Phase 1 (greedy)
    // uses it verbatim; the greedy phase takes DELTA=0.05.
    // regardless of window size.
    const double greedy_delta_d = delta_d;

    // ---- Step 1: Filter (P1-P5) — shared between greedy and bucket ----
    auto [flag, in_pruned, out_pruned, _sum_in_p, _sum_out_p] = check_rules_cpp(
        std::move(in_vec), std::move(out_vec),
        anchor_side_is_in, anchor_val, theta, eps, sum_in, sum_out
    );
    (void)_sum_in_p; (void)_sum_out_p;

    if (flag == 0) return py::none();

    // Anchor-alone wedge check (cheap; skip both phases if trigger alone works)
    double initial_sa, initial_sb;
    if (anchor_side_is_in) { initial_sa = anchor_val; initial_sb = 0.0; }
    else                   { initial_sa = 0.0;        initial_sb = anchor_val; }
    if (initial_sa >= theta) {
        const double g0 = std::abs(initial_sa - initial_sb);
        if (g0 <= eps * (1.0 + greedy_delta_d) * initial_sa) {
            return py::make_tuple(initial_sa, initial_sb);
        }
    }

    // ---- Step 2: Adaptive-max greedy (Phase 1) ----
    // Uses the original delta_d with no adaptive scaling.
    // adaptive_max_greedy_cpp mutates its vectors (pop_back). So copy the
    // pruned lists first so bucket (if needed) sees the original.
    std::vector<double> in_greedy  = in_pruned;
    std::vector<double> out_greedy = out_pruned;

    Result greedy_r = adaptive_max_greedy_cpp(
        in_greedy, out_greedy,
        anchor_side_is_in, anchor_val,
        theta, eps, greedy_delta_d
    );

    if (greedy_r.found) {
        // Greedy hit — no need for bucket
        return py::make_tuple(greedy_r.sa, greedy_r.sb);
    }

    // ---- Step 3: Bucket fallback (Phase 2) on the ORIGINAL pruned lists ----
    const int n_in_p  = static_cast<int>(in_pruned.size());
    const int n_out_p = static_cast<int>(out_pruned.size());
    const int n_p = n_in_p + n_out_p;
    if (n_p == 0) return py::none();

    // ---- Adaptive parameter scaling — uses POST-FILTER n (matches b14) ----
    // Rationale: the wedge-bucket fallback scales based on
    // len(candidates) which is *already pruned* by check_rules. Using the
    // pre-filter n here would coarsen the buckets unnecessarily and cause
    // a strict subset of witnesses to survive (78 asymmetric misses observed
    // on LI-Small full when we used pre-filter n).
    if (n_p > 3000) {
        delta_sa = std::max(delta_sa, 0.35);
        delta_d  = std::max(delta_d,  0.35);
        max_states = std::min(max_states, 300);
    } else if (n_p > 1500) {
        delta_sa = std::max(delta_sa, 0.25);
        delta_d  = std::max(delta_d,  0.25);
        max_states = std::min(max_states, 600);
    } else if (n_p > 500) {
        delta_sa = std::max(delta_sa, 0.18);
        delta_d  = std::max(delta_d,  0.18);
        max_states = std::min(max_states, 1200);
    } else if (n_p > 100) {
        delta_sa = std::max(delta_sa, 0.13);
        delta_d  = std::max(delta_d,  0.13);
        max_states = std::min(max_states, 2500);
    }

    // Combine + sort by amount DESCENDING (bucket's expected order)
    std::vector<std::pair<double, int8_t>> combined;
    combined.reserve(n_p);
    for (double a : in_pruned)  combined.emplace_back(a, static_cast<int8_t>(1));
    for (double a : out_pruned) combined.emplace_back(a, static_cast<int8_t>(0));
    std::sort(combined.begin(), combined.end(),
              [](const auto& x, const auto& y) { return x.first > y.first; });

    std::vector<double> cand_amts(n_p);
    std::vector<int8_t> cand_is_in(n_p);
    for (int i = 0; i < n_p; ++i) {
        cand_amts[i]  = combined[i].first;
        cand_is_in[i] = combined[i].second;
    }

    Result bucket_r = core_bucket_search_cpp(
        initial_sa, initial_sb,
        cand_amts, cand_is_in,
        theta, eps, delta_sa, delta_d, max_states, compression_mode
    );

    if (!bucket_r.found) return py::none();
    return py::make_tuple(bucket_r.sa, bucket_r.sb);
}

static py::object frontier_search_cascade_cpp(
        py::array_t<double, py::array::c_style | py::array::forcecast> in_amts,
        py::array_t<double, py::array::c_style | py::array::forcecast> out_amts,
        double anchor_val,
        std::string anchor_side,
        double theta, double eps,
        double delta_sa,
        double delta_d,
        int max_states) {
    return frontier_search_cascade_impl(
        in_amts, out_amts, anchor_val, anchor_side,
        theta, eps, delta_sa, delta_d, max_states,
        CompressionMode::LogMD
    );
}

static py::object frontier_search_cascade_compression_cpp(
        py::array_t<double, py::array::c_style | py::array::forcecast> in_amts,
        py::array_t<double, py::array::c_style | py::array::forcecast> out_amts,
        double anchor_val,
        std::string anchor_side,
        double theta, double eps,
        double delta_sa,
        double delta_d,
        int max_states,
        std::string compression_mode) {
    return frontier_search_cascade_impl(
        in_amts, out_amts, anchor_val, anchor_side,
        theta, eps, delta_sa, delta_d, max_states,
        parse_compression_mode(compression_mode)
    );
}


// =========================================================================
// pybind11 module
// =========================================================================
PYBIND11_MODULE(wedge_bucket_cpp, m) {
    m.doc() = "C++17 implementation of wedge feasibility algorithms. "
              "Exports entry points for full log-bucket search, adaptive-max "
              "greedy, and the full filter+greedy+bucket cascade.";

    m.def("frontier_search_full_cpp", &frontier_search_full_cpp,
          py::arg("in_amts"),
          py::arg("out_amts"),
          py::arg("anchor_val"),
          py::arg("anchor_side"),
          py::arg("theta"),
          py::arg("eps"),
          py::arg("delta_sa") = 0.1,
          py::arg("delta_d")  = 0.1,
          py::arg("max_states") = 4000,
          "Filter + bucket search in one call. Returns None or (sa, sb).");

    m.def("frontier_search_greedyonly_cpp", &frontier_search_greedyonly_cpp,
          py::arg("in_amts"),
          py::arg("out_amts"),
          py::arg("anchor_val"),
          py::arg("anchor_side"),
          py::arg("theta"),
          py::arg("eps"),
          py::arg("delta_d") = 0.05,
          "Filter + adaptive-max greedy (no bucket). Returns None or (sa, sb).");

    m.def("frontier_search_cascade_cpp", &frontier_search_cascade_cpp,
          py::arg("in_amts"),
          py::arg("out_amts"),
          py::arg("anchor_val"),
          py::arg("anchor_side"),
          py::arg("theta"),
          py::arg("eps"),
          py::arg("delta_sa") = 0.1,
          py::arg("delta_d")  = 0.1,
          py::arg("max_states") = 4000,
          "Filter + adaptive-max greedy + bucket fallback (matches "
          "the GeoWedge cascade). Returns None or (sa, sb).");

    m.def("frontier_search_cascade_compression_cpp",
          &frontier_search_cascade_compression_cpp,
          py::arg("in_amts"),
          py::arg("out_amts"),
          py::arg("anchor_val"),
          py::arg("anchor_side"),
          py::arg("theta"),
          py::arg("eps"),
          py::arg("delta_sa") = 0.1,
          py::arg("delta_d")  = 0.1,
          py::arg("max_states") = 4000,
          py::arg("compression_mode") = "log_md",
          "Filter + adaptive-max greedy + selectable bucket fallback. "
          "compression_mode in {log_md, uniform_mout, uniform_d}. "
          "Returns None or (sa, sb).");
}
