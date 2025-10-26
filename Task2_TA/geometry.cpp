#include <pybind11/pybind11.h>
#include <string>
#include <cmath>

namespace py = pybind11;

class Vector2D {
public:
    double x, y;
    // TODO: Implement your constructors and methods
    // Constructor
    Vector2D(double x, double y) : x(x), y(y) {}

    // Calculate the length of the vector
    double length() const {
        return std::sqrt(x * x + y * y);
    }

    // Add two vectors
    Vector2D add(const Vector2D& other) const {
        return Vector2D(x + other.x, y + other.y);
    }

    // Calculate the dot product
    double dot(const Vector2D& other) const {
        return x * other.x + y * other.y;
    }

    // String representation
    std::string toString() const {
        return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
    }

};

PYBIND11_MODULE(geometry, m) {
    m.doc() = "2D Geometry module with Vector2D class and operator overloading";
    py::class_<Vector2D>(m, "Vector2D")
    // TODO: Bind your attributes and methods
    //  .def_readwrite("attribute_name", &ClassName::attribute_name)
    //  .def("method_name", &ClassName::method_name)
        .def(py::init<double, double>(), 
             py::arg("x"), py::arg("y"),
             "Construct a 2D vector")
        .def_readwrite("x", &Vector2D::x, "X coordinate")
        .def_readwrite("y", &Vector2D::y, "Y coordinate")
        .def("length", &Vector2D::length, 
             "Calculate the magnitude of the vector")
        .def("add", &Vector2D::add, 
             py::arg("other"),
             "Add another vector to this vector")
        .def("dot", &Vector2D::dot, 
             py::arg("other"),
             "Calculate dot product with another vector")
        .def("toString", &Vector2D::toString,
             "Get string representation of the vector");
}

// Compile with:
// c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) geometry.cpp -o geometry$(python3-config --extension-suffix)