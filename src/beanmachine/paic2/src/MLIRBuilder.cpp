//
// Created by Steffi Stumpos on 7/15/22.
//
#include "MLIRBuilder.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRefOps.h.inc"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"

#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include <vector>

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace py = pybind11;
namespace mlir {
    class MLIRContext;
    class ModuleOp;
}

class MLIRGenImpl {
public:
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

    mlir::ModuleOp generate_op(std::shared_ptr<paic2::PythonModule> pythonModule) {
        theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

        for (std::shared_ptr <paic2::PythonFunction> f: pythonModule->getFunctions()) {
            generate_op(f);
        }

        if (failed(mlir::verify(theModule))) {
            theModule.emitError("module verification error");
            return nullptr;
        }
        return theModule;
    }

private:
    mlir::ModuleOp theModule;
    mlir::OpBuilder builder;

    mlir::Location loc(const paic2::Location &loc) {
        return mlir::FileLineColLoc::get(builder.getStringAttr("file_not_implemented_yet"), loc.getLine(),
                                         loc.getCol());
    }

    mlir::func::FuncOp generate_op(std::shared_ptr<paic2::PythonFunction> &pythonFunction) {
        builder.setInsertionPointToEnd(theModule.getBody());
        std::vector<mlir::Type> arg_types;
        mlir::TypeRange inputs(llvm::makeArrayRef(arg_types));
        auto funcType = builder.getFunctionType({}, {});
        auto location = loc(pythonFunction->loc());
        mlir::func::FuncOp func_op = builder.create<mlir::func::FuncOp>(location, pythonFunction->getName(), funcType);
        if(func_op.empty()){
            func_op.addEntryBlock();
        }

        mlir::Block &entryBlock = func_op.front();
        builder.setInsertionPointToStart(&entryBlock);
        builder.create<mlir::func::ReturnOp>(loc(pythonFunction->loc()));
        return func_op;
    }
};

void paic2::MLIRBuilder::bind(py::module &m) {
    py::class_<MLIRBuilder>(m, "MLIRBuilder")
            .def(py::init<py::object>(), py::arg("context") = py::none())
            .def("print_func_name", &MLIRBuilder::print_func_name);
}

paic2::MLIRBuilder::MLIRBuilder(pybind11::object contextObj) {}

void paic2::MLIRBuilder::print_func_name(std::shared_ptr<paic2::PythonFunction> function) {
    ::mlir::MLIRContext *context = new ::mlir::MLIRContext();
    context->loadDialect<mlir::func::FuncDialect>();
    std::vector<std::shared_ptr<PythonFunction>> funcs;
    funcs.push_back(function);
    auto py_module = std::make_shared<paic2::PythonModule>(funcs);
    mlir::ModuleOp module = MLIRGenImpl(*context).generate_op(py_module);
    module.dump();
}