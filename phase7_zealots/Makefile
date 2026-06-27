# Reuses xoshiro.h from phase3 via the include path (-I), so we don't duplicate it.
CXX = g++
CXXFLAGS = -std=c++20 -march=native -O3 -Wall -I../phase3_cpp

mc_zealots: mc_zealots.cpp ../phase3_cpp/xoshiro.h
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f mc_zealots
