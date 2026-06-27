# The single C++ engine for the whole project. C++20, aggressive optimisation.
CXX = g++
CXXFLAGS = -std=c++20 -march=native -O3 -Wall

mc_engine: mc_engine.cpp xoshiro.h
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f mc_engine
