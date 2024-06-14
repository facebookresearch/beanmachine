//
// Created by Steffi Stumpos on 6/10/22.
//
#include <iostream>

#include "MLIRBuilder.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/IR/Module.h"

#include <numeric>
#include <string>

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
    template <typename OpTy>
    class OwningOpRef;
    class ModuleOp;
} // namespace mlir

PYBIND11_MODULE(paic_mlir, m) {
    m.doc() = "MVP for pybind module";
    paic_mlir::MLIRBuilder::bind(m);
    paic_mlir::Node::bind(m);
}

void paic_mlir::MLIRBuilder::bind(py::module &m) {
    py::class_<MLIRBuilder>(m, "MLIRBuilder")
            .def(py::init<py::object>(), py::arg("context") = py::none())
            .def("to_metal", &MLIRBuilder::to_metal);
}

paic_mlir::MLIRBuilder::MLIRBuilder(pybind11::object contextObj) {}

namespace {
    class MLIRGenImpl {
    public:
        MLIRGenImpl(mlir::MLIRContext &context) : builder(&context){}

        mlir::ModuleOp generate_op(std::shared_ptr<paic_mlir::PythonModule> pythonModule) {
            // We create an empty MLIR module and codegen functions one at a time and
            // add them to the module.
            theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

            for (std::shared_ptr<paic_mlir::PythonFunction> f : pythonModule->getFunctions()){
                generate_op(f);
            }

            // Verify the module after we have finished constructing it, this will check
            // the structural properties of the IR and invoke any specific verifiers we
            // have on the Toy operations.
            if (failed(mlir::verify(theModule))) {
                theModule.emitError("module verification error");
                return nullptr;
            }
            return theModule;
        }
    private:
        mlir::ModuleOp theModule;
        mlir::OpBuilder builder;
        llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
        llvm::ScopedHashTable<llvm::StringRef, mlir::func::FuncOp> functionSymbolTable;

        mlir::Location loc(const paic_mlir::Location &loc) {
            return mlir::FileLineColLoc::get(builder.getStringAttr("file_not_implemented_yet"), loc.getLine(),
                                             loc.getCol());
        }

        mlir::Type getTensorType(ArrayRef<int64_t> shape) {
            if (shape.empty())
                return mlir::UnrankedTensorType::get(builder.getF64Type());
            return mlir::RankedTensorType::get(shape, builder.getF64Type());
        }

        mlir::Type getType(const paic_mlir::Type type) {
            if(std::strcmp(type.getName().data(), "float") == 0){
                return builder.getF32Type();
            } else {
                // TODO: insert appropriate Not Implemented exception
                throw 0;
            }
        }

        mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
            if (symbolTable.count(var))
                return mlir::failure();
            symbolTable.insert(var, value);
            return mlir::success();
        }

        mlir::LogicalResult declare(llvm::StringRef var, mlir::func::FuncOp value) {
            if (functionSymbolTable.count(var))
                return mlir::failure();
            functionSymbolTable.insert(var, value);
            return mlir::success();
        }

        mlir::Value mlirGen(std::shared_ptr<paic_mlir::GetValNode> expr) {
            if (auto variable = symbolTable.lookup(expr->getName()))
                return variable;
            emitError(loc(expr->loc()), "error: unknown variable '")
                    << expr->getName() << "'";
            return nullptr;
        }

        mlir::Value mlirGen(paic_mlir::GetValNode* expr) {
            if (auto variable = symbolTable.lookup(expr->getName()))
                return variable;

            emitError(loc(expr->loc()), "error: unknown variable '")
                    << expr->getName() << "'";
            return nullptr;
        }

        mlir::LogicalResult mlirGen(paic_mlir::ReturnNode* ret) {
            auto location = loc(ret->loc());
            mlir::Value expr = nullptr;
            if (ret->getValue()) {
                if (!(expr = mlirGen(ret->getValue().get())))
                    return mlir::failure();
            }
            builder.create<mlir::func::ReturnOp>(location, expr ? makeArrayRef(expr) : ArrayRef<mlir::Value>());
            return mlir::success();
        }

        mlir::Value mlirGen(paic_mlir::CallNode* call) {
            llvm::StringRef callee = call->getCallee();
            auto location = loc(call->loc());

            // Codegen the operands first.
            SmallVector<mlir::Value, 4> operands;
            for (std::shared_ptr<paic_mlir::Expression> expr : call->getArgs()) {
                auto arg = mlirGen(expr.get());
                if (!arg)
                    return nullptr;
                operands.push_back(arg);
            }

            // Builtin calls have their custom operation, meaning this is a
            // straightforward emission.
            if (callee == "times") {
                if (call->getArgs().size() != 2) {
                    emitError(location, "MLIR codegen encountered an error: times "
                                        "accepts exactly two arguments");
                    return nullptr;
                }

                return builder.create<mlir::arith::MulFOp>(location, operands[0], operands[1]);
            } else if (callee == "math.pow") {
                return builder.create<mlir::math::PowFOp>(location, operands[0], operands[1]);
            }

            // Otherwise this is a call to a user-defined function. Calls to
            // user-defined functions are mapped to a custom call that takes the callee
            // name as an attribute.
            // arguments: FuncOp callee, ValueRange operands = {}
            if (mlir::func::FuncOp functionOp = functionSymbolTable.lookup(callee)){
                auto call = builder.create<mlir::func::CallOp>(location, functionOp, operands);
                // TODO: discover correct abstraction for values
                emitError(location, "User defined functions not supported yet");
                return nullptr;
            } else {
                emitError(location, "Unreconized function :" + callee);
                return nullptr;
            }
        }

        mlir::Value mlirGen(paic_mlir::ConstNode<float>* expr) {
            //auto type = getType(expr->getType());
            return builder.create<mlir::arith::ConstantFloatOp>(loc(expr->loc()), llvm::APFloat(expr->getValue()), builder.getF32Type());
        }

        mlir::Value mlirGen(paic_mlir::Expression* expr) {
            switch (expr->getKind()) {
                case paic_mlir::NodeKind::GetVal:
                    return mlirGen(dynamic_cast<paic_mlir::GetValNode*>(expr));
                case paic_mlir::NodeKind::Constant:
                    // TODO: cast to ConstNode parent and query primitive type
                    return mlirGen(dynamic_cast<paic_mlir::ConstNode<float>*>(expr));
                case paic_mlir::NodeKind::Call:
                    return mlirGen(dynamic_cast<paic_mlir::CallNode*>(expr));
                default:
                    emitError(loc(expr->loc()))
                            << "MLIR codegen encountered an unhandled expr kind '"
                            << Twine(expr->getKind()) << "'";
                    return nullptr;
            }
        }
        mlir::Value mlirGen(paic_mlir::VarNode* vardecl) {
            std::shared_ptr<paic_mlir::Expression> init = vardecl->getInitVal();
            if (!init) {
                emitError(loc(vardecl->loc()),"missing initializer in variable declaration");
                return nullptr;
            }
            mlir::Value value = mlirGen(init.get());
            if (!value)
                return nullptr;
            if (failed(declare(vardecl->getName(), value)))
                return nullptr;
            return value;
        }
        /// Codegen a list of expression, return failure if one of them hit an error.
        mlir::LogicalResult mlirGen(std::shared_ptr<paic_mlir::BlockNode> blockNode) {
            ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
            for (std::shared_ptr<paic_mlir::Node> expr : blockNode->getChildren()) {
                // Specific handling for variable declarations, return statement, and
                // print. These can only appear in block list and not in nested
                // expressions.
                if (auto *var = dyn_cast<paic_mlir::VarNode>(expr.get())) {
                    if (!mlirGen(var))
                        return mlir::failure();
                    continue;
                }
                if (auto *var = dyn_cast<paic_mlir::ReturnNode>(expr.get())) {
                    if (mlir::failed(mlirGen(var)))
                        return mlir::failure();
                    continue;
                }
            }
            return mlir::success();
        }

        mlir::func::FuncOp generate_op(std::shared_ptr<paic_mlir::PythonFunction> &pythonFunction) {
            // Create a scope in the symbol table to hold variable declarations.
            ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

            // Create an MLIR function for the given prototype.
            builder.setInsertionPointToEnd(theModule.getBody());
            // create FuncOp
            auto location = loc(pythonFunction->loc());

            // TODO: change the number of inlined elements in small array from 4?
            int i=0;
            std::vector<mlir::Type> arg_types(pythonFunction->getArgs().size());
            for(std::shared_ptr<paic_mlir::ParamNode> p : pythonFunction->getArgs()){
                auto type = getType(p->getType());
                arg_types[i++] = type;
            }

            // create a function using the Func dialect
            mlir::TypeRange inputs(llvm::makeArrayRef(arg_types));
            mlir::FunctionType funcType = builder.getFunctionType(inputs, getType(pythonFunction->getType()));
            // TODO: add attributes here if relevant
            //mlir::Attr
           // attribute array looks like this: ::llvm::ArrayRef<::mlir::NamedAttribute>{emit_llvm_attr})
            mlir::func::FuncOp func_op = builder.create<mlir::func::FuncOp>(location, pythonFunction->getName(), funcType);
            func_op->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(func_op->getContext()));
            func_op.addEntryBlock();
            mlir::Block &entryBlock = func_op.front();
            auto protoArgs = pythonFunction->getArgs();

            // Declare all the function arguments in the symbol table.
            for (const auto nameValue :
                    llvm::zip(protoArgs, entryBlock.getArguments())) {
                if (failed(declare(std::get<0>(nameValue)->getName(),
                                   std::get<1>(nameValue))))
                    return nullptr;
            }
            // Set the insertion point in the builder to the beginning of the function
            // body, it will be used throughout the codegen to create operations in this
            // function.
            builder.setInsertionPointToStart(&entryBlock);
            // Emit the body of the function.
            if (mlir::failed(mlirGen(pythonFunction->getBody()))) {
                func_op.erase();
                return nullptr;
            }
            mlir::func::ReturnOp returnOp;
            if (!entryBlock.empty())
                returnOp = dyn_cast<mlir::func::ReturnOp>(entryBlock.back());
            if (!returnOp) {
                builder.create<mlir::func::ReturnOp>(loc(pythonFunction->loc()));
            } else if (!returnOp.operands().empty()) {
                // Otherwise, if this return operation has an operand then add a result to
                // the function.
                func_op.setType(builder.getFunctionType(func_op.getFunctionType().getInputs(), getType(pythonFunction->getType())));
            }
            return func_op;
        }
    };
}

pybind11::float_ paic_mlir::MLIRBuilder::to_metal(std::shared_ptr<paic_mlir::PythonFunction> function, pybind11::float_ input) {
    // MLIR context (load any custom dialects you want to use)
    ::mlir::MLIRContext *context = new ::mlir::MLIRContext();
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::math::MathDialect>();
    context->loadDialect<mlir::arith::ArithmeticDialect>();
    mlir::registerAllDialects(*context);

    // MLIR Module. Create the module
    std::vector<std::shared_ptr<PythonFunction>> functions{ function };
    std::shared_ptr<PythonModule> py_module = std::make_shared<PythonModule>(functions);
    MLIRGenImpl generator(*context);
    auto mlir_module = generator.generate_op(py_module);
    mlir_module->dump();

    // todo: add passes and run module
    mlir::PassManager pm(context);
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::arith::createConvertArithmeticToLLVMPass());
    auto result = pm.run(mlir_module);

    // Lower to machine code
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(*context);

    // An optimization pipeline to use within the execution engine.
    auto optPipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/3, /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
    // the module.
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine = mlir::ExecutionEngine::create(mlir_module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    // Invoke the JIT-compiled function.
    float res = 0;
    auto invocationResult = engine->invoke(function->getName(), input.operator float(), mlir::ExecutionEngine::result(res));
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        throw 0;
    }
    return pybind11::float_(res);
}