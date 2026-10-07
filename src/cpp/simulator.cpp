// PitWallML Phase 4c: C++20 Monte Carlo race simulator, exposed to Python with pybind11.
//
// Two layers:
//   * pure C++ "core" functions working on raw pointers (no Python types): the
//     number crunching;
//   * pybind11 wrappers that check shapes, get pointers to the NumPy buffers WITHOUT
//     copying them, release the GIL, call a core, and hand a NumPy array back.
//
// Both cores must reproduce the Python reference results exactly:
//   static_core   <-> src/models/baselines/monte_carlo.py::simulate_race_times
//   reactive_core <-> src/models/baselines/reactive.py::simulate_reactive_py

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

struct RaceConfig {
    int total_laps = 0;
    double base_lap_s = 0.0;
    std::vector<double> offsets;  // index = compound index (RaceParams.compounds order)
    std::vector<double> degs;
    double pit_loss_s = 0.0;
    double sc_lap_factor = 1.4;
    double sc_pit_loss_factor = 0.5;
    double fuel_effect_s_per_lap = 0.06;
};

struct ReactiveConfig {
    int start = 0;               // compound indices, like RaceConfig.offsets
    int second = 1;
    int plan_lap = 1;            // 1-based lap number of the planned in-lap
    int sc_window = 0;
    int extra_compound = -1;     // -1 = no extra stop
    int extra_min_age = 15;
    int extra_min_laps_left = 10;
};

// ---------------------------------------------------------------------------------
// Core functions: plain C++, no Python objects, safe to run without the GIL.
//
// Memory layout: sc_mask and noise are (n_sims x total_laps) NumPy arrays in C order
// (row-major), so element [s][l] (0-based lap index l) lives at s * total_laps + l.
// out has n_sims elements: the total race time of each simulation.
// ---------------------------------------------------------------------------------

namespace {

// One lap's time. Shared by both cores, so the fixed-strategy and the reactive
// simulator can never disagree about the physics. Mirrors the Python formula term by
// term, in the same order, so floating-point results match the reference exactly.
inline double lap_seconds(const RaceConfig& cfg, int compound, double age, int laps_remaining, bool sc,
                          double noise) {
    if (sc) return cfg.base_lap_s * cfg.sc_lap_factor;  // everyone slow: no tyre, fuel or noise effect
    return cfg.base_lap_s + cfg.offsets[compound] + cfg.degs[compound] * age +
           cfg.fuel_effect_s_per_lap * laps_remaining + noise;
}

// Extra time for driving through the pit lane on an in-lap (half price under the SC).
inline double pit_seconds(const RaceConfig& cfg, bool sc) {
    return cfg.pit_loss_s * (sc ? cfg.sc_pit_loss_factor : 1.0);
}

}  // namespace

void static_core(const RaceConfig& cfg,
                 const std::int64_t* compound_idx,  // (total_laps,)
                 const double* tyre_age,            // (total_laps,)
                 const bool* pit,                   // (total_laps,)
                 const bool* sc_mask,               // (n_sims, total_laps)
                 const double* noise,               // (n_sims, total_laps)
                 std::ptrdiff_t n_sims,
                 double* out) {
    const int L = cfg.total_laps;
    for (std::ptrdiff_t s = 0; s < n_sims; ++s) {
        const bool* sc_row = sc_mask + s * L;  // this simulation's row: L consecutive values
        const double* noise_row = noise + s * L;
        double total = 0.0;
        for (int l = 0; l < L; ++l) {
            const bool sc = sc_row[l];
            const int c = static_cast<int>(compound_idx[l]);
            total += lap_seconds(cfg, c, tyre_age[l], L - (l + 1), sc, noise_row[l]);
            if (pit[l]) total += pit_seconds(cfg, sc);
        }
        out[s] = total;
    }
}

void reactive_core(const RaceConfig& cfg,
                   const ReactiveConfig& pol,
                   const bool* sc_mask,     // (n_sims, total_laps)
                   const double* noise,     // (n_sims, total_laps)
                   std::ptrdiff_t n_sims,
                   double* out) {
    const int L = cfg.total_laps;
    const bool has_extra = pol.extra_compound >= 0;  // -1 encodes Python's None
    for (std::ptrdiff_t s = 0; s < n_sims; ++s) {
        const bool* sc_row = sc_mask + s * L;
        const double* noise_row = noise + s * L;

        // Per-simulation state: reset for every simulated race.
        int compound = pol.start;
        int age = 1;
        bool stopped = false;
        bool extra_done = false;
        double total = 0.0;

        for (int lap = 1; lap <= L; ++lap) {  // 1-based, exactly like the Python reference
            const bool sc = sc_row[lap - 1];
            total += lap_seconds(cfg, compound, age, L - lap, sc, noise_row[lap - 1]);

            int next = -1;  // compound to switch to after this lap; -1 = stay out
            if (!stopped) {
                const bool in_window = pol.plan_lap - pol.sc_window <= lap && lap < pol.plan_lap;
                if (lap == pol.plan_lap || (sc && in_window)) {
                    next = pol.second;
                    stopped = true;
                }
            } else if (has_extra && !extra_done && sc && age >= pol.extra_min_age &&
                       L - lap >= pol.extra_min_laps_left) {
                next = pol.extra_compound;
                extra_done = true;
            }

            if (next >= 0) {
                total += pit_seconds(cfg, sc);
                compound = next;
                age = 1;
            } else {
                ++age;
            }
        }
        out[s] = total;
    }
}

// ---------------------------------------------------------------------------------
// pybind11 layer: Python types in, checks, zero-copy pointers out.
//
// py::array_t<T, c_style | forcecast>: if the caller passes a C-ordered array of the
// right dtype, pybind11 hands us a view of THE SAME memory (zero copy). Anything else
// (wrong dtype, Fortran order, a list) is converted into a temporary copy first.
// ---------------------------------------------------------------------------------

using BoolArray = py::array_t<bool, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

namespace {

void check_config(const RaceConfig& cfg) {
    if (cfg.total_laps < 1) throw std::invalid_argument("total_laps must be >= 1");
    if (cfg.offsets.empty() || cfg.offsets.size() != cfg.degs.size())
        throw std::invalid_argument("offsets and degs must be non-empty and the same length");
}

std::ptrdiff_t check_random_arrays(const RaceConfig& cfg, const BoolArray& sc_mask, const DoubleArray& noise) {
    if (sc_mask.ndim() != 2 || noise.ndim() != 2 || sc_mask.shape(0) != noise.shape(0) ||
        sc_mask.shape(1) != cfg.total_laps || noise.shape(1) != cfg.total_laps)
        throw std::invalid_argument("sc_mask and noise must both have shape (n_sims, " +
                                    std::to_string(cfg.total_laps) + ")");
    return sc_mask.shape(0);
}

void check_policy(const RaceConfig& cfg, const ReactiveConfig& pol) {
    const int n = static_cast<int>(cfg.offsets.size());
    auto valid = [n](int c) { return 0 <= c && c < n; };
    if (!valid(pol.start) || !valid(pol.second) || pol.start == pol.second)
        throw std::invalid_argument("start/second must be different valid compound indices");
    if (pol.extra_compound >= n) throw std::invalid_argument("extra_compound out of range");
    if (pol.plan_lap < 1 || pol.plan_lap >= cfg.total_laps)
        throw std::invalid_argument("plan_lap must be in 1..total_laps-1");
    if (pol.sc_window < 0 || pol.extra_min_age < 1 || pol.extra_min_laps_left < 1)
        throw std::invalid_argument("sc_window >= 0, extra_min_age >= 1, extra_min_laps_left >= 1");
}

}  // namespace

py::array_t<double> simulate_static(const RaceConfig& cfg, const IntArray& compound_idx, const DoubleArray& tyre_age,
                                    const BoolArray& pit, const BoolArray& sc_mask, const DoubleArray& noise) {
    check_config(cfg);
    for (const auto* arr : {static_cast<const py::array*>(&compound_idx), static_cast<const py::array*>(&tyre_age),
                            static_cast<const py::array*>(&pit)})
        if (arr->ndim() != 1 || arr->shape(0) != cfg.total_laps)
            throw std::invalid_argument("compound_idx, tyre_age and pit must have shape (total_laps,)");
    for (py::ssize_t l = 0; l < cfg.total_laps; ++l)
        if (compound_idx.at(l) < 0 || compound_idx.at(l) >= static_cast<std::int64_t>(cfg.offsets.size()))
            throw std::invalid_argument("compound_idx out of range");
    const std::ptrdiff_t n_sims = check_random_arrays(cfg, sc_mask, noise);

    py::array_t<double> totals(n_sims);
    double* out = totals.mutable_data();
    {
        py::gil_scoped_release release;  // pure C++ from here on: other Python threads may run
        static_core(cfg, compound_idx.data(), tyre_age.data(), pit.data(), sc_mask.data(), noise.data(), n_sims, out);
    }
    return totals;
}

py::array_t<double> simulate_reactive(const RaceConfig& cfg, const ReactiveConfig& pol, const BoolArray& sc_mask,
                                      const DoubleArray& noise) {
    check_config(cfg);
    check_policy(cfg, pol);
    const std::ptrdiff_t n_sims = check_random_arrays(cfg, sc_mask, noise);
    py::array_t<double> totals(n_sims);
    double* out = totals.mutable_data();
    {
        py::gil_scoped_release release;
        reactive_core(cfg, pol, sc_mask.data(), noise.data(), n_sims, out);
    }
    return totals;
}

py::array_t<double> simulate_reactive_batch(const RaceConfig& cfg, const std::vector<ReactiveConfig>& policies,
                                            const BoolArray& sc_mask, const DoubleArray& noise) {
    check_config(cfg);
    for (const auto& pol : policies) check_policy(cfg, pol);
    const std::ptrdiff_t n_sims = check_random_arrays(cfg, sc_mask, noise);
    const auto n_pol = static_cast<py::ssize_t>(policies.size());
    py::array_t<double> totals({n_pol, static_cast<py::ssize_t>(n_sims)});
    double* out = totals.mutable_data();
    {
        py::gil_scoped_release release;  // one call for the whole grid: no Python overhead per policy
        for (std::size_t p = 0; p < policies.size(); ++p)
            reactive_core(cfg, policies[p], sc_mask.data(), noise.data(), n_sims, out + p * n_sims);
    }
    return totals;
}

PYBIND11_MODULE(_pitwall_sim, m) {
    m.doc() = "PitWallML C++20 Monte Carlo race simulator";

    py::class_<RaceConfig>(m, "RaceConfig")
        .def(py::init<>())
        .def_readwrite("total_laps", &RaceConfig::total_laps)
        .def_readwrite("base_lap_s", &RaceConfig::base_lap_s)
        .def_readwrite("offsets", &RaceConfig::offsets)
        .def_readwrite("degs", &RaceConfig::degs)
        .def_readwrite("pit_loss_s", &RaceConfig::pit_loss_s)
        .def_readwrite("sc_lap_factor", &RaceConfig::sc_lap_factor)
        .def_readwrite("sc_pit_loss_factor", &RaceConfig::sc_pit_loss_factor)
        .def_readwrite("fuel_effect_s_per_lap", &RaceConfig::fuel_effect_s_per_lap);

    py::class_<ReactiveConfig>(m, "ReactiveConfig")
        .def(py::init<>())
        .def_readwrite("start", &ReactiveConfig::start)
        .def_readwrite("second", &ReactiveConfig::second)
        .def_readwrite("plan_lap", &ReactiveConfig::plan_lap)
        .def_readwrite("sc_window", &ReactiveConfig::sc_window)
        .def_readwrite("extra_compound", &ReactiveConfig::extra_compound)
        .def_readwrite("extra_min_age", &ReactiveConfig::extra_min_age)
        .def_readwrite("extra_min_laps_left", &ReactiveConfig::extra_min_laps_left);

    m.def("simulate_static", &simulate_static, py::arg("cfg"), py::arg("compound_idx"), py::arg("tyre_age"),
          py::arg("pit"), py::arg("sc_mask"), py::arg("noise"),
          "Total race time per simulation of a fixed strategy, shape (n_sims,).");
    m.def("simulate_reactive", &simulate_reactive, py::arg("cfg"), py::arg("policy"), py::arg("sc_mask"),
          py::arg("noise"), "Total race time per simulation of a reactive policy, shape (n_sims,).");
    m.def("simulate_reactive_batch", &simulate_reactive_batch, py::arg("cfg"), py::arg("policies"),
          py::arg("sc_mask"), py::arg("noise"),
          "Race times of many reactive policies on the same races, shape (n_policies, n_sims).");
}
