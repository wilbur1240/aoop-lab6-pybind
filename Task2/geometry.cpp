#include <pybind11/pybind11.h>
#include <string>
#include <cmath>

namespace py = pybind11;

class Vector2D {
public:
    double x, y;
    // TODO: Implement your constructors and methods
    // Constructor

    // methods
    double length() const {

    }

    Vector2D add(const Vector2D& other) const {

    }

    double dot(const Vector2D& other) const {

    }

    std::string toString() const {
        
    }
};

PYBIND11_MODULE(geometry, m) {
    py::class_<Vector2D>(m, "Vector2D")
        .def(py::init<double, double>(),
             py::arg("x"), py::arg("y"),
             "Construct a 2D vector")
    // TODO: Bind your attributes and methods
    // .def_readwrite("attribute_name", &ClassName::attribute_name)
    // .def("method_name", &ClassName::method_name)
}   

// Compile with:
// c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) geometry.cpp -o geometry$(python3-config --extension-suffix)