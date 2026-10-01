// C++17 core routines for the baselines.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace py = pybind11;

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

static std::vector<double> to_vector(Array arr) {
    const auto n = static_cast<std::size_t>(arr.size());
    return std::vector<double>(arr.data(), arr.data() + n);
}

static bool feasible(double sa, double sb, double theta, double eps) {
    return sa >= theta && std::abs(sa - sb) <= eps * sa;
}

static double ratio_score(double amount, double trigger) {
    if (amount <= 0.0 || trigger <= 0.0) return 0.0;
    return std::min(amount, trigger) / std::max(amount, trigger);
}

static int enumerate_subsets(const std::vector<double>& in_amts,
                             const std::vector<double>& out_amts,
                             double initial_sa, double initial_sb,
                             double theta, double eps) {
    const std::size_t n_in = in_amts.size();
    const std::size_t n_out = out_amts.size();
    if (n_in >= 62 || n_out >= 62) {
        throw std::runtime_error("top-k enumeration received too many values");
    }

    const std::uint64_t masks_in = 1ULL << n_in;
    const std::uint64_t masks_out = 1ULL << n_out;
    std::vector<double> sum_in(masks_in, 0.0);
    std::vector<double> sum_out(masks_out, 0.0);

    for (std::uint64_t mask = 1; mask < masks_in; ++mask) {
        const std::uint64_t bit = mask & (~mask + 1);
        const int idx = __builtin_ctzll(bit);
        sum_in[mask] = sum_in[mask ^ bit] + in_amts[static_cast<std::size_t>(idx)];
    }
    for (std::uint64_t mask = 1; mask < masks_out; ++mask) {
        const std::uint64_t bit = mask & (~mask + 1);
        const int idx = __builtin_ctzll(bit);
        sum_out[mask] = sum_out[mask ^ bit] + out_amts[static_cast<std::size_t>(idx)];
    }

    for (double add_in : sum_in) {
        const double sa = initial_sa + add_in;
        if (sa < theta) continue;
        for (double add_out : sum_out) {
            if (feasible(sa, initial_sb + add_out, theta, eps)) return 1;
        }
    }
    return 0;
}

static void sort_desc(std::vector<double>& values) {
    std::sort(values.begin(), values.end(), std::greater<double>());
}

static std::vector<double> topk_by_value(Array arr, int k) {
    auto values = to_vector(arr);
    sort_desc(values);
    if (k >= 0 && static_cast<std::size_t>(k) < values.size()) {
        values.resize(static_cast<std::size_t>(k));
    }
    return values;
}

static std::vector<double> topk_by_ratio(Array arr, double trigger, int k) {
    auto values = to_vector(arr);
    std::sort(values.begin(), values.end(), [trigger](double a, double b) {
        const double sa = ratio_score(a, trigger);
        const double sb = ratio_score(b, trigger);
        if (sa != sb) return sa > sb;
        return a > b;
    });
    if (k >= 0 && static_cast<std::size_t>(k) < values.size()) {
        values.resize(static_cast<std::size_t>(k));
    }
    return values;
}

static int query_topk_value(Array in_arr, Array out_arr,
                             double trigger, const std::string& anchor_type,
                             double theta, double eps, int k) {
    auto in_amts = topk_by_value(in_arr, k);
    auto out_amts = topk_by_value(out_arr, k);
    const double initial_sa = anchor_type == "in" ? trigger : 0.0;
    const double initial_sb = anchor_type == "in" ? 0.0 : trigger;
    return enumerate_subsets(in_amts, out_amts, initial_sa, initial_sb, theta, eps);
}

static int query_topk_ratio(Array in_arr, Array out_arr,
                             double trigger, const std::string& anchor_type,
                             double theta, double eps, int k) {
    auto in_amts = topk_by_ratio(in_arr, trigger, k);
    auto out_amts = topk_by_ratio(out_arr, trigger, k);
    const double initial_sa = anchor_type == "in" ? trigger : 0.0;
    const double initial_sb = anchor_type == "in" ? 0.0 : trigger;
    return enumerate_subsets(in_amts, out_amts, initial_sa, initial_sb, theta, eps);
}

static int greedy_alternating(std::vector<double> in_amts,
                              std::vector<double> out_amts,
                              double trigger, const std::string& anchor_type,
                              double theta, double eps) {
    double sa = anchor_type == "in" ? trigger : 0.0;
    double sb = anchor_type == "in" ? 0.0 : trigger;
    if (feasible(sa, sb, theta, eps)) return 1;

    std::size_t i_in = 0, i_out = 0;
    while (i_in < in_amts.size() || i_out < out_amts.size()) {
        if (sa <= sb) {
            if (i_in < in_amts.size()) sa += in_amts[i_in++];
            else if (i_out < out_amts.size()) sb += out_amts[i_out++];
            else break;
        } else {
            if (i_out < out_amts.size()) sb += out_amts[i_out++];
            else if (i_in < in_amts.size()) sa += in_amts[i_in++];
            else break;
        }
        if (feasible(sa, sb, theta, eps)) return 1;
    }
    return 0;
}

static int query_greedy_value(Array in_arr, Array out_arr,
                               double trigger, const std::string& anchor_type,
                               double theta, double eps) {
    auto in_amts = to_vector(in_arr);
    auto out_amts = to_vector(out_arr);
    sort_desc(in_amts);
    sort_desc(out_amts);
    return greedy_alternating(std::move(in_amts), std::move(out_amts),
                              trigger, anchor_type, theta, eps);
}

static int query_greedy_ratio(Array in_arr, Array out_arr,
                               double trigger, const std::string& anchor_type,
                               double theta, double eps) {
    auto in_amts = topk_by_ratio(in_arr, trigger, -1);
    auto out_amts = topk_by_ratio(out_arr, trigger, -1);
    return greedy_alternating(std::move(in_amts), std::move(out_amts),
                              trigger, anchor_type, theta, eps);
}

static int query_greedy_fill(Array in_arr, Array out_arr,
                              double trigger, const std::string& anchor_type,
                              double theta, double eps) {
    auto in_amts = to_vector(in_arr);
    auto out_amts = to_vector(out_arr);
    sort_desc(in_amts);
    std::sort(out_amts.begin(), out_amts.end());

    double sa = anchor_type == "in" ? trigger : 0.0;
    double sb = anchor_type == "in" ? 0.0 : trigger;
    if (feasible(sa, sb, theta, eps)) return 1;

    for (double amount : in_amts) {
        if (sa >= theta) break;
        sa += amount;
        if (feasible(sa, sb, theta, eps)) return 1;
    }
    if (sa < theta) return 0;

    std::vector<double> remaining = std::move(out_amts);
    while (!remaining.empty()) {
        const double target = sa - sb;
        if (target <= 0.0) break;
        auto it = std::lower_bound(remaining.begin(), remaining.end(), target);
        int best_idx = -1;
        double best_diff = std::abs(sa - sb);

        auto consider = [&](std::vector<double>::iterator cand) {
            if (cand == remaining.end()) return;
            const double new_diff = std::abs(sa - (sb + *cand));
            if (new_diff < best_diff) {
                best_diff = new_diff;
                best_idx = static_cast<int>(cand - remaining.begin());
            }
        };
        if (it != remaining.begin()) consider(std::prev(it));
        consider(it);

        if (best_idx < 0) break;
        sb += remaining[static_cast<std::size_t>(best_idx)];
        remaining.erase(remaining.begin() + best_idx);
        if (feasible(sa, sb, theta, eps)) return 1;
    }
    return 0;
}

PYBIND11_MODULE(baseline_cpp_core, m) {
    m.doc() = "C++17 core query routines for the baselines.";
    m.def("query_topk_value", &query_topk_value);
    m.def("query_topk_ratio", &query_topk_ratio);
    m.def("query_greedy_value", &query_greedy_value);
    m.def("query_greedy_ratio", &query_greedy_ratio);
    m.def("query_greedy_fill", &query_greedy_fill);
}
