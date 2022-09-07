//
// Created by Steffi Stumpos on 7/15/22.
//
#include "PaicAST.h"
#include <pybind11/stl.h>
namespace py = pybind11;

void paic2::PythonModule::bind(pybind11::module &m) {
    py::class_<Location>(m, "Location").def(py::init<int,int>());
    py::class_<PythonFunction, std::shared_ptr<PythonFunction>>(m, "PythonFunction")
            .def(py::init<Location, const std::string &>());
}