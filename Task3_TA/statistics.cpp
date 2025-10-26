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
    DataAnalyzer() {}
    
    // Add a single value
    void addValue(double value) {
        data.push_back(value);
    }
    
    // Add multiple values
    void addValues(const std::vector<double>& values) {
        data.insert(data.end(), values.begin(), values.end());
    }
    
    // Calculate mean (average)
    double getMean() const {
        if (data.empty()) {
            throw std::runtime_error("Cannot calculate mean of empty dataset");
        }
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        return sum / data.size();
    }
    
    // Get minimum value
    double getMin() const {
        if (data.empty()) {
            throw std::runtime_error("Cannot get min of empty dataset");
        }
        return *std::min_element(data.begin(), data.end());
    }
    
    // Get maximum value
    double getMax() const {
        if (data.empty()) {
            throw std::runtime_error("Cannot get max of empty dataset");
        }
        return *std::max_element(data.begin(), data.end());
    }
    
    // Get count of values
    int getCount() const {
        return static_cast<int>(data.size());
    }
    
    // Get all values
    std::vector<double> getValues() const {
        return data;
    }
    
private:
    std::vector<double> data;
};

PYBIND11_MODULE(statistics, m) {
    py::class_<DataAnalyzer>(m, "DataAnalyzer")
    // TODO: Bind your methods
        .def(py::init<>(), "Construct a new DataAnalyzer")
        .def("addValue", &DataAnalyzer::addValue,
            py::arg("value"),
            "Add a single value to the dataset")
        .def("addValues", &DataAnalyzer::addValues,
            py::arg("values"),
            "Add multiple values to the dataset")
        .def("getMean", &DataAnalyzer::getMean,
            "Calculate the mean (average) of all values")
        .def("getMin", &DataAnalyzer::getMin,
            "Get the minimum value in the dataset")
        .def("getMax", &DataAnalyzer::getMax,
            "Get the maximum value in the dataset")
        .def("getCount", &DataAnalyzer::getCount,
            "Get the number of values in the dataset")
        .def("getValues", &DataAnalyzer::getValues,
            "Get all values in the dataset");
}

// Compile with:
// c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) statistics.cpp -o statistics$(python3-config --extension-suffix)