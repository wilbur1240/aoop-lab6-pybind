#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // STL container support
#include <vector>
#include <algorithm>
#include <numeric>

namespace py = pybind11;

class DataAnalyzer {
public:
    // TODO: Implement your methods
    // Constructor

    // methods
    void addValue(double value) {

    }

    void addValues(const std::vector<double>& values) {

    }

    double getMean() const {

    }

    double getMin() const {

    }

    double getMax() const {

    }

    int getCount() const {

    }

    std::vector<double> getValues() const {

    }
private:
    std::vector<double> data;
};

PYBIND11_MODULE(statistics, m) {
    py::class_<DataAnalyzer>(m, "DataAnalyzer")
        .def(py::init<>(), "Construct a new DataAnalyzer");
    // TODO: Bind your methods
}

// Compile with:
// c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) statistics.cpp -o statistics$(python3-config --extension-suffix)