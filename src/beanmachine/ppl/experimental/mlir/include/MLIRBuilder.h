//
// Created by Steffi Stumpos on 6/10/22.
//

#ifndef PAIC_IR_MLIRBUILDER_H
#define PAIC_IR_MLIRBUILDER_H

#include <pybind11/pybind11.h>
#include "mlir-c/IR.h"
#include "PaicAST.h"

namespace paic_mlir {
    class MLIRBuilder {
    public:
        static void bind(pybind11::module &m);
        MLIRBuilder(pybind11::object contextObj);
        pybind11::float_ to_metal(std::shared_ptr<paic_mlir::PythonFunction> function, pybind11::float_ input);
    };
}

#endif //PAIC_IR_MLIRBUILDER_H
