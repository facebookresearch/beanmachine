//
// Created by Steffi Stumpos on 7/20/22.
//
#include "bm/BMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::bm;

#include "bm/bm_dialect.cpp.inc"

#define GET_OP_CLASSES
#include "bm/bm_ops.cpp.inc"

void BMDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "bm/bm_ops.cpp.inc"
    >();
    addTypes<WorldType>();
}

//===----------------------------------------------------------------------===//
// BM Types
//===----------------------------------------------------------------------===//

namespace mlir {
    namespace bm {

        struct WorldTypeStorage : public mlir::TypeStorage {
            /// The `KeyTy` is a required type that provides an interface for the storage
            /// instance. This type will be used when uniquing an instance of the type
            /// storage. For our struct type, we will unique each instance structurally on
            /// the elements that it contains.
            using KeyTy = llvm::ArrayRef<mlir::Type>;

            /// A constructor for the type storage instance.
            WorldTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
                    : elementTypes(elementTypes) {}

            /// Define the comparison function for the key type with the current storage
            /// instance. This is used when constructing a new instance to ensure that we
            /// haven't already uniqued an instance of the given key.
            bool operator==(const KeyTy &key) const { return key == elementTypes; }

            /// Define a hash function for the key type. This is used when uniquing
            /// instances of the storage, see the `WorldType::get` method.
            /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
            /// have hash functions available, so we could just omit this entirely.
            static llvm::hash_code hashKey(const KeyTy &key) {
                return llvm::hash_value(key);
            }

            /// Define a construction function for the key type from a set of parameters.
            /// These parameters will be provided when constructing the storage instance
            /// itself.
            /// Note: This method isn't necessary because KeyTy can be directly
            /// constructed with the given parameters.
            static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
                return KeyTy(elementTypes);
            }

            /// Define a construction method for creating a new instance of this storage.
            /// This method takes an instance of a storage allocator, and an instance of a
            /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
            /// allocations used to create the type storage and its internal.
            static WorldTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                               const KeyTy &key) {
                // Copy the elements from the provided `KeyTy` into the allocator.
                llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

                // Allocate the storage instance and construct it.
                return new(allocator.allocate<WorldTypeStorage>())
                        WorldTypeStorage(elementTypes);
            }

            /// The following field contains the element types of the struct.
            llvm::ArrayRef<mlir::Type> elementTypes;
        };

    }
}

/// Create an instance of a `WorldType` with the given element types. There
/// *must* be at least one element type.
WorldType WorldType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.empty() && "expected at least 1 element type");

    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. The first parameter is the context to unique in. The
    // parameters after the context are forwarded to the storage instance.
    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx, elementTypes);
}

/// Returns the element types of this struct type.
llvm::ArrayRef<mlir::Type> WorldType::getElementTypes() {
    // 'getImpl' returns a pointer to the internal storage instance.
    return getImpl()->elementTypes;
}

/// Parse an instance of a type registered to the toy dialect.
mlir::Type BMDialect::parseType(mlir::DialectAsmParser &parser) const {
    // Parse a struct type in the following form:
    //   struct-type ::= `struct` `<` type (`,` type)* `>`

    // NOTE: All MLIR parser function return a ParseResult. This is a
    // specialization of LogicalResult that auto-converts to a `true` boolean
    // value on failure to allow for chaining, but may be used with explicit
    // `mlir::failed/mlir::succeeded` as desired.

    // Parse: `struct` `<`
    if (parser.parseKeyword("struct") || parser.parseLess())
        return Type();

    // Parse the element types of the struct.
    SmallVector<mlir::Type, 1> elementTypes;
    do {
        // Parse the current element type.
        SMLoc typeLoc = parser.getCurrentLocation();
        mlir::Type elementType;
        if (parser.parseType(elementType))
            return nullptr;

        // Check that the type is either a TensorType or another WorldType.
        if (!elementType.isa<mlir::TensorType, WorldType>()) {
            parser.emitError(typeLoc, "element type for a struct must either "
                                      "be a TensorType or a WorldType, got: ")
                    << elementType;
            return Type();
        }
        elementTypes.push_back(elementType);

        // Parse the optional: `,`
    } while (succeeded(parser.parseOptionalComma()));

    // Parse: `>`
    if (parser.parseGreater())
        return Type();
    return WorldType::get(elementTypes);
}

/// Print an instance of a type registered to the toy dialect.
void BMDialect::printType(mlir::Type type,
                          mlir::DialectAsmPrinter &printer) const {
    // Currently the only bm type is a world type.
    WorldType structType = type.cast<WorldType>();

    // Print the struct type according to the parser format.
    printer << "world<";
    llvm::interleaveComma(structType.getElementTypes(), printer);
    printer << '>';
}


/// Verify that the given attribute value is valid for the given type.
static mlir::LogicalResult verifyConstantForType(mlir::Type type,
                                                 mlir::Attribute opaqueValue,
                                                 mlir::Operation *op) {
    if (type.isa<mlir::TensorType>()) {
        // Check that the value is an elements attribute.
        auto attrValue = opaqueValue.dyn_cast<mlir::DenseFPElementsAttr>();
        if (!attrValue)
            return op->emitError("constant of TensorType must be initialized by "
                                 "a DenseFPElementsAttr, got ")
                    << opaqueValue;

        // If the return type of the constant is not an unranked tensor, the shape
        // must match the shape of the attribute holding the data.
        auto resultType = type.dyn_cast<mlir::RankedTensorType>();
        if (!resultType)
            return success();

        // Check that the rank of the attribute type matches the rank of the
        // constant result type.
        auto attrType = attrValue.getType().cast<mlir::TensorType>();
        if (attrType.getRank() != resultType.getRank()) {
            return op->emitOpError("return type must match the one of the attached "
                                   "value attribute: ")
                    << attrType.getRank() << " != " << resultType.getRank();
        }

        // Check that each of the dimensions match between the two types.
        for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
            if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
                return op->emitOpError(
                        "return type shape mismatches its attribute at dimension ")
                        << dim << ": " << attrType.getShape()[dim]
                        << " != " << resultType.getShape()[dim];
            }
        }
        return mlir::success();
    }
    auto resultType = type.cast<WorldType>();
    llvm::ArrayRef<mlir::Type> resultElementTypes = resultType.getElementTypes();

    // Verify that the initializer is an Array.
    auto attrValue = opaqueValue.dyn_cast<ArrayAttr>();
    if (!attrValue || attrValue.getValue().size() != resultElementTypes.size())
        return op->emitError("constant of WorldType must be initialized by an "
                             "ArrayAttr with the same number of elements, got ")
                << opaqueValue;

    // Check that each of the elements are valid.
    llvm::ArrayRef<mlir::Attribute> attrElementValues = attrValue.getValue();
    for (const auto it : llvm::zip(resultElementTypes, attrElementValues))
        if (failed(verifyConstantForType(std::get<0>(it), std::get<1>(it), op)))
            return mlir::failure();
    return mlir::success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void mlir::bm::FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                             llvm::StringRef name, mlir::FunctionType type,
                             llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    // FunctionOpInterface provides a convenient `build` method that will populate
    // the state of our FuncOp, and create an entry block.
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult mlir::bm::FuncOp::parse(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
    // Dispatch to the FunctionOpInterface provided utility method that parses the
    // function operation.
    auto buildFuncType =
            [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
               llvm::ArrayRef<mlir::Type> results,
               mlir::function_interface_impl::VariadicFlag,
               std::string &) { return builder.getFunctionType(argTypes, results); };

    return mlir::function_interface_impl::parseFunctionOp(
            parser, result, /*allowVariadic=*/false, buildFuncType);
}

void mlir::bm::FuncOp::print(mlir::OpAsmPrinter &p) {
    // Dispatch to the FunctionOpInterface provided utility method that prints the
    // function operation.
    mlir::function_interface_impl::printFunctionOp(p, *this,
            /*isVariadic=*/false);
}
/// Returns the region on the function operation that is callable.
mlir::Region *FuncOp::getCallableRegion() { return &getBody(); }

/// Returns the results types that the callable region produces when
/// executed.
llvm::ArrayRef<mlir::Type> FuncOp::getCallableResults() {
    return getFunctionType().getResults();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::bm::ReturnOp::verify() {
    // We know that the parent operation is a function, because of the 'HasParent'
    // trait attached to the operation definition.
    auto function = cast<FuncOp>((*this)->getParentOp());

    /// ReturnOps can only have a single optional operand.
    if (getNumOperands() > 1)
        return emitOpError() << "expects at most 1 return operand";

    // The operand number and types must match the function signature.
    const auto &results = function.getFunctionType().getResults();
    if (getNumOperands() != results.size())
        return emitOpError() << "does not return the same number of values ("
                             << getNumOperands() << ") as the enclosing function ("
                             << results.size() << ")";

    // If the operation does not have an input, we are done.
    if (!hasOperand())
        return mlir::success();

    auto inputType = *operand_type_begin();
    auto resultType = results.front();

    // Check that the result type of the function matches the operand type.
    if (inputType == resultType)
        return mlir::success();

    return emitError() << "type of return operand (" << inputType
                       << ") doesn't match function result type (" << resultType
                       << ")";
}