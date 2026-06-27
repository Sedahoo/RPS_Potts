// Unified Monte Carlo engine for the Potts-RPS-on-networks model.
//
// This is THE simulation driver for the whole project (the original repo's
// "DRIVERS/" idea): a single binary the experiment scripts call. With the
// default --zealot-frac 0 it is the plain agent-level MC (phases 2-6); with
// zealots enabled it runs the stubborn-node experiments (phase 7).
//
// Zealots: a fraction of nodes are permanently locked to strategy `zs`. They
// are skipped in the update loop but still influence their neighbours. Placement
// is random or on the highest-degree nodes (--zealot-target hub). We also report
// the CONVERSION RATE: of the FREE (non-zealot) nodes, what fraction play zs --
// separating genuine influence from the zealots' own trivial contribution.
//
// Output line:  "m_psi avg_r avg_p avg_s conversion_zs"
// Build:  make

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <numeric>
#include <algorithm>
#include "xoshiro.h"

struct Graph { int n = 0; std::vector<std::vector<int>> adj; };

Graph read_graph(const std::string& path) {
    std::ifstream f(path);
    Graph g;
    if (!f.is_open()) return g;
    std::map<int,int> id; int next = 0;
    auto idx = [&](int v){ auto it=id.find(v); if(it!=id.end()) return it->second;
                           id[v]=next; return next++; };
    int u, v;
    while (f >> u >> v) {
        int a = idx(u), b = idx(v);
        if ((int)g.adj.size() <= std::max(a,b)) g.adj.resize(std::max(a,b)+1);
        g.adj[a].push_back(b); g.adj[b].push_back(a);
    }
    g.n = next;
    return g;
}

int main(int argc, char* argv[]) {
    std::string graph_path, output_path, zealot_target = "random";
    double epsilon = 0.5, temp = 0.65, zealot_frac = 0.0, zealot_frac_b = 0.0;
    int sweeps = 1500, burn_in = 450, zealot_strategy = 0, zealot_strategy_b = 1;
    unsigned long seed = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a=="--graph"   && i+1<argc) graph_path  = argv[++i];
        else if (a=="--epsilon" && i+1<argc) epsilon     = std::stod(argv[++i]);
        else if (a=="--temp"    && i+1<argc) temp        = std::stod(argv[++i]);
        else if (a=="--sweeps"  && i+1<argc) sweeps      = std::stoi(argv[++i]);
        else if (a=="--burn-in" && i+1<argc) burn_in     = std::stoi(argv[++i]);
        else if (a=="--seed"    && i+1<argc) seed        = std::stoul(argv[++i]);
        else if (a=="--zealot-frac"     && i+1<argc) zealot_frac     = std::stod(argv[++i]);
        else if (a=="--zealot-strategy" && i+1<argc) zealot_strategy = std::stoi(argv[++i]);
        else if (a=="--zealot-target"   && i+1<argc) zealot_target   = argv[++i];  // random | hub
        else if (a=="--zealot-frac-b"     && i+1<argc) zealot_frac_b     = std::stod(argv[++i]);  // 2nd faction
        else if (a=="--zealot-strategy-b" && i+1<argc) zealot_strategy_b = std::stoi(argv[++i]);
        else if (a=="--output"  && i+1<argc) output_path = argv[++i];
    }
    if (graph_path.empty()) { std::cerr << "need --graph\n"; return 1; }
    Graph g = read_graph(graph_path);
    if (g.n == 0) { std::cerr << "empty graph\n"; return 1; }

    xso::xoshiro_8x64_star_star rng(seed);
    const double P[3][3] = {{1.0,-epsilon,epsilon},{epsilon,1.0,-epsilon},{-epsilon,epsilon,1.0}};

    std::vector<int> state(g.n);
    for (int i = 0; i < g.n; ++i) state[i] = rng.sample(0, 2);

    // pick the zealots, all locked to `zealot_strategy`.
    //   target=random : a uniformly random subset of nodes.
    //   target=hub    : the highest-degree nodes (tests whether hubs, which BA
    //                   graphs have and ER graphs don't, amplify zealot power).
    std::vector<char> is_zealot(g.n, 0);
    int n_zealot = (int)std::lround(zealot_frac * g.n);
    std::vector<int> order(g.n);
    std::iota(order.begin(), order.end(), 0);
    if (zealot_target == "hub") {
        std::sort(order.begin(), order.end(),
                  [&](int x, int y){ return g.adj[x].size() > g.adj[y].size(); });
        for (int i = 0; i < n_zealot; ++i) {
            is_zealot[order[i]] = 1;
            state[order[i]] = zealot_strategy;
        }
    } else {
        for (int i = 0; i < n_zealot; ++i) {             // partial Fisher-Yates
            int j = rng.sample(i, g.n - 1);
            std::swap(order[i], order[j]);
            is_zealot[order[i]] = 1;
            state[order[i]] = zealot_strategy;
        }
    }
    // optional second zealot faction, drawn at random from the remaining nodes
    int n_zealot_b = (int)std::lround(zealot_frac_b * g.n);
    for (int i = n_zealot; i < n_zealot + n_zealot_b && i < g.n; ++i) {
        int j = rng.sample(i, g.n - 1);
        std::swap(order[i], order[j]);
        is_zealot[order[i]] = 1;
        state[order[i]] = zealot_strategy_b;
    }
    int n_free = g.n - n_zealot - n_zealot_b;

    const double sin120 = std::sqrt(3.0) / 2.0;
    double sr=0,sp=0,ss=0,spr=0,spi=0,conv=0;
    long meas = 0;

    for (int t = 0; t < sweeps; ++t) {
        for (int i = 0; i < g.n; ++i) {
            int n = rng.sample(0, g.n - 1);
            if (is_zealot[n]) continue;                  // zealots never update
            int cur = state[n];
            int prop = (cur + rng.sample(1, 2)) % 3;
            const auto& nb = g.adj[n];
            if (nb.empty()) continue;
            double dU = 0.0;
            for (int m : nb) dU += P[prop][state[m]] - P[cur][state[m]];
            if (rng.sample(0.0, 1.0) * (1.0 + std::exp(-dU/temp)) < 1.0)
                state[n] = prop;
        }
        if (t >= burn_in) {
            int c[3] = {0,0,0}; long free_zs = 0;
            for (int i = 0; i < g.n; ++i) {
                c[state[i]]++;
                if (!is_zealot[i] && state[i] == zealot_strategy) free_zs++;
            }
            double r=(double)c[0]/g.n, p=(double)c[1]/g.n, s=(double)c[2]/g.n;
            sr+=r; sp+=p; ss+=s;
            spr += r - 0.5*(p+s);
            spi += sin120*(p - s);
            conv += (n_free > 0) ? (double)free_zs / n_free : 0.0;
            meas++;
        }
    }
    double m_psi = std::sqrt(std::pow(spr/meas,2) + std::pow(spi/meas,2));
    std::ostringstream line;
    line << m_psi << " " << sr/meas << " " << sp/meas << " " << ss/meas
         << " " << conv/meas << "\n";
    if (output_path.empty()) std::cout << line.str();
    else { std::ofstream o(output_path); o << line.str(); }
    return 0;
}
