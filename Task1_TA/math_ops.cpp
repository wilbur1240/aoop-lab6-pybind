#include <pybind11/pybind11.h>

namespace py = pybind11;

// TODO: Implement your math operations here
// Function 1: add two integers
int add(int a, int b) {
    return a + b;
}

// Function 2: multiply two doubles
double multiply(double a, double b) {
    return a * b;
}

// Function 3: calculate factorial
int factorial(int n) {
    if (n < 0) return -1; // Error for negative numbers
    if (n == 0 || n == 1) return 1;
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

PYBIND11_MODULE(math_ops, m) {
    m.doc() = "Basic math operations module";

    // TODO: Bind your functions here
    // Example: m.def("add", &add, "A function that adds two integers");
    m.def("add", &add, "Add two integers",
        py::arg("a"), py::arg("b"));

    m.def("multiply", &multiply, "Multiply two doubles",
        py::arg("a"), py::arg("b"));

    m.def("factorial", &factorial, "Calculate factorial of n",
        py::arg("n"));
}

// Compile with:
// c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) math_ops.cpp -o math_ops$(python3-config --extension-suffix)