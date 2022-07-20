//
// Created by Steffi Stumpos on 7/15/22.
//
#include <pybind11/pybind11.h>
#include "MLIRBuilder.h"
#include "PaicAST.h"

PYBIND11_MODULE(paic2, m) {
    m.doc() = "PAIC2 module";
    paic2::MLIRBuilder::bind(m);
    paic2::Node::bind(m);
}