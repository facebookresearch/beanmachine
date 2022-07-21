//
// Created by Steffi Stumpos on 7/15/22.
//

#ifndef PAIC2_MLIRBUILDER_H
#define PAIC2_MLIRBUILDER_H

#include <pybind11/pybind11.h>
#include "mlir-c/IR.h"
#include "PaicAST.h"
#include "WorldSpec.h"

namespace paic2 {

    class MLIRBuilder {
    public:
        static void bind(pybind11::module &m);
        MLIRBuilder(pybind11::object contextObj);
        MLIRBuilder(){}
        void print_func_name(std::shared_ptr<paic2::PythonFunction> function);
        void infer(std::shared_ptr<paic2::PythonFunction> function, paic2::WorldSpec const& worldClassSpec);
        pybind11::float_ evaluate(std::shared_ptr<paic2::PythonFunction> function, pybind11::float_ input);
    };
}

#endif //PAIC2_MLIRBUILDER_H
