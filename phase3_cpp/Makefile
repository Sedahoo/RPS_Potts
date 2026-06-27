# Same flags the original repo uses: C++20, aggressive optimisation, tuned to
# this machine. -march=native is what makes the hot loop fly.
CXX = g++
CXXFLAGS = -std=c++20 -march=native -O3 -Wall

mc_simulation: mc_simulation.cpp xoshiro.h
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f mc_simulation
