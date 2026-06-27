// Phase 3: the Phase-2 Monte Carlo, ported to C++ for speed.
//
// Identical physics to phase2_mc/mc.py -- same payoff matrix, same Glauber
// acceptance, same order parameter -- but ~50-100x faster, so we can afford
// the big (k, epsilon) parameter sweeps that Phase 4 needs.
//
// This mirrors the original repo's mc_simulation.cpp. Key ingredients that make
// it fast: the xoshiro PRNG (far faster + better than std::mt19937), -O3
// -march=native, and a flat adjacency structure.
//
// Build:  make
// Run:    ./mc_simulation --graph g.edgelist --epsilon 0.5 --temp 0.65 --sweeps 1500 --burn-in 450 --seed 1
// Output: one line  "m_psi avg_r avg_p avg_s"  (to stdout, or --output file)

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <array>
#include "xoshiro.h"

struct Graph { int n = 0; std::vector<std::vector<int>> adj; };

// Read an undirected edgelist ("u v" per line), remapping arbitrary node ids
// to 0..n-1. Same logic as the original repo.
Graph read_graph(const std::string& path) {
    std::ifstream f(path);
    Graph g;
    if (!f.is_open()) return g;
    std::map<int,int> id;
    int next = 0;
    auto idx = [&](int v){ auto it = id.find(v); if (it!=id.end()) return it->second;
                           id[v]=next; return next++; };
    int u, v;
    while (f >> u >> v) {
        int a = idx(u), b = idx(v);
        if ((int)g.adj.size() <= std::max(a,b)) g.adj.resize(std::max(a,b)+1);
        g.adj[a].push_back(b);
        g.adj[b].push_back(a);
    }
    g.n = next;
    return g;
}

int main(int argc, char* argv[]) {
    std::string graph_path, output_path;
    double epsilon = 0.5, temp = 0.65;
    int sweeps = 1500, burn_in = 450;
    unsigned long seed = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--graph"   && i+1 < argc) graph_path  = argv[++i];
        else if (a == "--epsilon" && i+1 < argc) epsilon     = std::stod(argv[++i]);
        else if (a == "--temp"    && i+1 < argc) temp        = std::stod(argv[++i]);
        else if (a == "--sweeps"  && i+1 < argc) sweeps      = std::stoi(argv[++i]);
        else if (a == "--burn-in" && i+1 < argc) burn_in     = std::stoi(argv[++i]);
        else if (a == "--seed"    && i+1 < argc) seed        = std::stoul(argv[++i]);
        else if (a == "--output"  && i+1 < argc) output_path = argv[++i];
    }
    if (graph_path.empty()) { std::cerr << "need --graph\n"; return 1; }

    Graph g = read_graph(graph_path);
    if (g.n == 0) { std::cerr << "empty graph\n"; return 1; }

    xso::xoshiro_8x64_star_star rng(seed);

    // payoff matrix P = I + eps*skew
    const double P[3][3] = {{1.0,-epsilon,epsilon},{epsilon,1.0,-epsilon},{-epsilon,epsilon,1.0}};

    std::vector<int> state(g.n);
    for (int i = 0; i < g.n; ++i) state[i] = rng.sample(0, 2);

    const double sin120 = std::sqrt(3.0) / 2.0;
    double sum_r=0, sum_p=0, sum_s=0, sum_psi_re=0, sum_psi_im=0;
    long measurements = 0;

    for (int t = 0; t < sweeps; ++t) {
        for (int i = 0; i < g.n; ++i) {                 // one sweep = n attempts
            int n   = rng.sample(0, g.n - 1);
            int cur = state[n];
            int prop = (cur + rng.sample(1, 2)) % 3;     // one of the other two
            const auto& nbrs = g.adj[n];
            if (nbrs.empty()) continue;
            double dU = 0.0;
            for (int m : nbrs) dU += P[prop][state[m]] - P[cur][state[m]];
            // accept with probability logistic(dU/temp)
            if (rng.sample(0.0, 1.0) * (1.0 + std::exp(-dU/temp)) < 1.0)
                state[n] = prop;
        }
        if (t >= burn_in) {
            int c[3] = {0,0,0};
            for (int s : state) c[s]++;
            double r = (double)c[0]/g.n, p = (double)c[1]/g.n, s = (double)c[2]/g.n;
            sum_r += r; sum_p += p; sum_s += s;
            sum_psi_re += r - 0.5*(p+s);                  // real part of psi
            sum_psi_im += sin120*(p - s);                 // imag part of psi
            measurements++;
        }
    }

    double ar=sum_r/measurements, ap=sum_p/measurements, as=sum_s/measurements;
    double m_psi = std::sqrt(std::pow(sum_psi_re/measurements,2)
                           + std::pow(sum_psi_im/measurements,2));

    std::ostringstream line;
    line << m_psi << " " << ar << " " << ap << " " << as << "\n";
    if (output_path.empty()) std::cout << line.str();
    else { std::ofstream o(output_path); o << line.str(); }
    return 0;
}
