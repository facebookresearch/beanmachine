//
// Created by Steffi Stumpos on 7/15/22.
//
#include "MLIRBuilder.h"
#include "bm/passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Host.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"

#include "bm/BMDialect.h"
#include "pybind_utils.h"
#include <pybind11/stl_bind.h>

using Tensor = std::vector<double, std::allocator<double>>;
PYBIND11_MAKE_OPAQUE(Tensor);

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

class MLIRGenImpl {
public:
    MLIRGenImpl(mlir::MLIRContext &context, paic2::WorldSpec const& spec) : builder(&context), _world_spec(spec){}
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}
    mlir::ModuleOp generate_op(std::shared_ptr<paic2::PythonModule> pythonModule) {
        // We create an empty MLIR module and codegen functions one at a time and
        // add them to the module.
        theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

        for (std::shared_ptr<paic2::PythonFunction> f : pythonModule->getFunctions()){
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
    paic2::WorldSpec _world_spec;
    llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

    mlir::Location loc(const paic2::Location &loc) {
        return mlir::FileLineColLoc::get(builder.getStringAttr("file_not_implemented_yet"), loc.getLine(),
                                         loc.getCol());
    }

    mlir::Type typeForCode(paic2::PrimitiveCode code){
        switch(code){
            case paic2::PrimitiveCode::Double:
                return builder.getF64Type();
            case paic2::PrimitiveCode::Float:
                return builder.getF32Type();
            case paic2::PrimitiveCode::Void:
                return builder.getNoneType();
            default:
                throw std::invalid_argument("unrecognized primitive type");
        }
    }

    // There are two types supported: PrimitiveType and the builtin type WorldType
    // A World type is compiled into a composite type that contains an array, sets of function pointers, and metadata
    mlir::Type getType(std::shared_ptr<paic2::Type> type) {
        if(auto *var = dyn_cast<paic2::PrimitiveType>(type.get())){
            return typeForCode(var->code());
        } else if (auto *var = dyn_cast<paic2::WorldType>(type.get())) {
            std::vector<mlir::Type> members;
            mlir::Type elementType = typeForCode(var->nodeType());
            ArrayRef<int64_t> shape(var->length());
            auto dataType = mlir::RankedTensorType::get(shape, elementType);
            members.push_back(dataType);
            mlir::Type structType = mlir::bm::WorldType::get(members);
            return structType;
        }
        throw std::invalid_argument("unrecognized primitive type");
    }

    mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
        if (symbolTable.count(var))
            return mlir::failure();
        symbolTable.insert(var, value);
        return mlir::success();
    }

    mlir::Value mlirGen(paic2::GetValNode* expr) {
        if (auto variable = symbolTable.lookup(expr->getName()))
            return variable;

        emitError(loc(expr->loc()), "error: unknown variable '")
                << expr->getName() << "'";
        return nullptr;
    }

    mlir::LogicalResult mlirGen(paic2::ReturnNode* ret) {
        auto location = loc(ret->loc());
        mlir::Value expr = nullptr;
        if (ret->getValue()) {
            if (!(expr = mlirGen(ret->getValue().get())))
                return mlir::failure();
        }
        builder.create<mlir::bm::ReturnOp>(location, expr ? makeArrayRef(expr) : ArrayRef<mlir::Value>());
        return mlir::success();
    }

    mlir::Value mlirGen(paic2::CallNode* call) {
        llvm::StringRef callee = call->getCallee();
        auto location = loc(call->loc());

        // Codegen the operands first.
        SmallVector<mlir::Value, 4> operands;
        for (std::shared_ptr<paic2::Expression> expr : call->getArgs()) {
            auto arg = mlirGen(expr.get());
            if (!arg)
                return nullptr;
            operands.push_back(arg);
        }

        // TODO: make map and remove magic strings
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

        // TODO
        emitError(location, "User defined functions not supported yet");
        return nullptr;
    }

    mlir::OpState opFromCall(paic2::CallNode* call) {
        llvm::StringRef callee = call->getCallee();
        auto location = loc(call->loc());
        SmallVector<mlir::Value, 4> operands;
        for (std::shared_ptr<paic2::Expression> expr : call->getArgs()) {
            auto arg = mlirGen(expr.get());
            if (!arg)
                throw std::invalid_argument("invalid argument during expression generation");
            operands.push_back(arg);
        }

        mlir::Value receiver(nullptr);
        if(call->getReceiver().get() != nullptr){
            receiver = mlirGen(call->getReceiver().get());
        }
        if(strcmp(callee.data(), _world_spec.print_name().data()) == 0){
            // we add an operator here so that all the operations relevant to printing are kept
            // inside a single operation until we are ready to lower to llvm ir
            if(receiver != nullptr && receiver.getType().isa<mlir::bm::WorldType>()){
                mlir::bm::PrintWorldOp result =  builder.create<mlir::bm::PrintWorldOp>(location, receiver);
                return result;
            } else {
                throw std::invalid_argument("only world print operations are supported");
            }
        } else {
            throw std::invalid_argument("only world print operations are supported");
        }
    }

    mlir::Value mlirGen(paic2::Expression* expr) {
        switch (expr->getKind()) {
            case paic2::NodeKind::GetVal:
                return mlirGen(dynamic_cast<paic2::GetValNode*>(expr));
            case paic2::NodeKind::Constant:
            {
                auto const_type = dynamic_cast<paic2::FloatConstNode*>(expr);
                auto primitive_type = dynamic_cast<paic2::PrimitiveType*>(expr->getType().get());
                switch(primitive_type->code()){
                    case paic2::PrimitiveCode::Float:
                        return builder.create<mlir::arith::ConstantFloatOp>(loc(expr->loc()), llvm::APFloat(const_type->getValue()), builder.getF32Type());
                    default:
                        throw std::invalid_argument("only float primitives are supported now");
                }
            }
            case paic2::NodeKind::Call:
                return mlirGen(dynamic_cast<paic2::CallNode*>(expr));
            default:
                emitError(loc(expr->loc()))
                        << "MLIR codegen encountered an unhandled expr kind '"
                        << Twine(expr->getKind()) << "'";
                return nullptr;
        }
    }

    mlir::Value mlirGen(paic2::VarNode* vardecl) {
        std::shared_ptr<paic2::Expression> init = vardecl->getInitVal();
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

    mlir::LogicalResult mlirGen(std::shared_ptr<paic2::BlockNode> blockNode) {
        ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
        for (std::shared_ptr<paic2::Node> expr : blockNode->getChildren()) {
            if (auto *var = dyn_cast<paic2::VarNode>(expr.get())) {
                if (!mlirGen(var))
                    return mlir::failure();
                continue;
            }
            if (auto *var = dyn_cast<paic2::CallNode>(expr.get())) {
                if (!opFromCall(var))
                    return mlir::failure();
                continue;
            }
            if (auto *var = dyn_cast<paic2::ReturnNode>(expr.get())) {
                if (mlir::failed(mlirGen(var)))
                    return mlir::failure();
                continue;
            }
        }
        return mlir::success();
    }

    mlir::bm::FuncOp generate_op(std::shared_ptr<paic2::PythonFunction> &pythonFunction) {
        ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);
        builder.setInsertionPointToEnd(theModule.getBody());
        auto location = loc(pythonFunction->loc());

        int i=0;
        std::vector<mlir::Type> arg_types(pythonFunction->getArgs().size());
        for(std::shared_ptr<paic2::ParamNode> p : pythonFunction->getArgs()){
            auto type = getType(p->getType());
            arg_types[i++] = type;
        }

        mlir::TypeRange inputs(llvm::makeArrayRef(arg_types));

        mlir::FunctionType funcType;
        auto returnType = getType(pythonFunction->getType());
        if(returnType == nullptr || returnType.getTypeID() == builder.getNoneType().getTypeID()){
            funcType = builder.getFunctionType(inputs, {});
        } else {
            funcType = builder.getFunctionType(inputs, returnType);
        }
        mlir::bm::FuncOp func_op = builder.create<mlir::bm::FuncOp>(location, pythonFunction->getName(), funcType);
        func_op->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(func_op->getContext()));
        if(func_op.empty()){
            func_op.addEntryBlock();
        }

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
        mlir::bm::ReturnOp returnOp;
        if (!entryBlock.empty())
            returnOp = dyn_cast<mlir::bm::ReturnOp>(entryBlock.back());
        if (!returnOp) {
            builder.create<mlir::bm::ReturnOp>(loc(pythonFunction->loc()));
        } else if (!returnOp.hasOperand()) {
            // Otherwise, if this return operation has an operand then add a result to
            // the function.
            auto returnType = getType(pythonFunction->getType());
            if(returnType == nullptr){
                func_op.setType(builder.getFunctionType(func_op.getFunctionType().getInputs(), {}));
            } else {
                func_op.setType(builder.getFunctionType(func_op.getFunctionType().getInputs(), returnType));
            }

        }
        return func_op;
    }
};

void paic2::MLIRBuilder::bind(py::module &m) {
    py::class_<paic2::WorldSpec>(m, "WorldSpec")
            .def(py::init())
            .def("set_print_name", &WorldSpec::set_print_name)
            .def("set_world_size", &WorldSpec::set_world_size)
            .def("set_world_name", &WorldSpec::set_world_name);

    py::class_<MLIRBuilder>(m, "MLIRBuilder")
            .def(py::init<py::object>(), py::arg("context") = py::none())
            .def("print_func_name", &MLIRBuilder::print_func_name)
            .def("infer", &MLIRBuilder::infer)
            .def("evaluate", &MLIRBuilder::evaluate);
    bind_vector<double>(m, "Tensor", true);
}

paic2::MLIRBuilder::MLIRBuilder(pybind11::object contextObj) {}

std::shared_ptr<mlir::ExecutionEngine> execution_engine_for_function(std::shared_ptr<paic2::PythonFunction> function, paic2::WorldSpec const& worldClassSpec) {
    // transform python to MLIR
    std::string function_name = function->getName().data();
    std::vector<std::shared_ptr<paic2::PythonFunction>> functions{ function };
    std::shared_ptr<paic2::PythonModule> py_module = std::make_shared<paic2::PythonModule>(functions);
    ::mlir::MLIRContext *context = new ::mlir::MLIRContext();
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::bm::BMDialect>();
    context->loadDialect<mlir::math::MathDialect>();
    context->loadDialect<mlir::arith::ArithmeticDialect>();
    context->loadDialect<mlir::memref::MemRefDialect>();
    MLIRGenImpl generator(*context, worldClassSpec);
    mlir::ModuleOp mlir_module = generator.generate_op(py_module);
    mlir_module->dump();

    // lower to LLVM dialect
    mlir::PassManager pm(context);
    pm.addPass(mlir::bm::createLowerToFuncPass());
    pm.addPass(mlir::bm::createLowerToLLVMPass());
    auto result = pm.run(mlir_module);

    // prepare environment for machine code generation
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(*(mlir_module->getContext()));

    auto optPipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/0, /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> maybeEngine = mlir::ExecutionEngine::create(mlir_module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();
    auto ptr = std::move(engine);
    mlir_module->dump();
    return ptr;
}

void paic2::MLIRBuilder::print_func_name(std::shared_ptr<paic2::PythonFunction> function) {
    paic2::WorldSpec spec;
    spec.set_print_name("print");
    auto engine = execution_engine_for_function(function, spec);
}

void paic2::MLIRBuilder::infer(std::shared_ptr<paic2::PythonFunction> function, paic2::WorldSpec const& worldClassSpec, std::shared_ptr<std::vector<double>> init_nodes) {
    // Invoke the JIT-compiled function.
    auto engine = execution_engine_for_function(function, worldClassSpec);
    double* front = &(init_nodes->front());
    double* data_ptr = &(init_nodes->front());
    std::vector<double*> values(2);
    values[0] = front;
    values[1] = front;

    auto invocationResult = engine->invoke(function->getName().data(), values, data_ptr, 0, 0, 1);
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
    }
}

pybind11::float_ paic2::MLIRBuilder::evaluate(std::shared_ptr<paic2::PythonFunction> function, pybind11::float_ input) {
    paic2::WorldSpec spec;
    auto engine = execution_engine_for_function(function, spec);

    // Invoke the JIT-compiled function.
    float res = 0;
    auto invocationResult = engine->invoke(function->getName(), input.operator float(), mlir::ExecutionEngine::result(res));
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        throw 0;
    }
    return pybind11::float_(res);
}
