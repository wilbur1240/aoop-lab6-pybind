#include <pybind11/pybind11.h>

namespace py = pybind11;

// TODO: Implement your math operations here
int add(int a, int b) {

}

double multiply(double a, double b) {

}

int factorial(int n) {
    
}


PYBIND11_MODULE(math_ops, m) {
    m.doc() = "Basic math operations module";

    // TODO: Bind your functions here
    // Example: 
    // m.def("add", &add, "A function that adds two integers",
    //     py::arg("a"), py::arg("b"));
}

// Compile with:
// c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) math_ops.cpp -o math_ops$(python3-config --extension-suffix)