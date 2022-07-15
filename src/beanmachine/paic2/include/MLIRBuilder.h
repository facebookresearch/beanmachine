//
// Created by Steffi Stumpos on 7/15/22.
//

#ifndef PAIC2_MLIRBUILDER_H
#define PAIC2_MLIRBUILDER_H

#include <pybind11/pybind11.h>
#include "mlir-c/IR.h"
#include "PaicAST.h"

namespace paic2 {

    class MLIRBuilder {
    public:
        static void bind(pybind11::module &m);
        MLIRBuilder(pybind11::object contextObj);
        void print_func_name(std::shared_ptr<paic2::PythonFunction> function);
    };
}

#endif //PAIC2_MLIRBUILDER_H
