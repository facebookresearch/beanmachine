#ifndef BM_OPS
#define BM_OPS

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'bm' dialect in the ODS framework so that we
// can define our operations.
def BM_Dialect : Dialect {
  let name = "bm";
  let cppNamespace = "::mlir::bm";
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
  let useDefaultTypePrinterParser = 1;
}

// Base class for bm dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class BM_Op<string mnemonic, list<Trait> traits = []> :
Op<BM_Dialect, mnemonic, traits>;

// Provide a definition for the BM WorldType for use in ODS. This allows for
// using WorldType in a similar way to Tensor or MemRef. We use `DialectType`
// to demarcate the WorldType as belonging to the BM dialect.
def BM_WorldType :
DialectType<BM_Dialect, CPred<"$_self.isa<WorldType>()">,
        "BM World type">;

// this allows us to specify for example that a call operations has an out type of either BM_WorldType or F32.
// in reality it can also return a pointer but we are not there yet.
def BM_Type : AnyTypeOf<[F32, BM_WorldType]>;

def PrintWorldOp : BM_Op<"print_world"> {
    let summary = "print operation";
    let description = [{
    The "print" builtin operation prints a given world, and produces
        no results.
    }];

    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[BM_WorldType, F64MemRef, F32MemRef]>:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def FuncOp : BM_Op<"func", [
    DeclareOpInterfaceMethods<CallableOpInterface>, FunctionOpInterface,
    IsolatedFromAbove, Symbol
    ]> {
    let summary = "user defined function operation";
    let description = [{
    The "bm.func" operation represents a user defined function. These are
    callable SSA-region operations that contain toy computations.

    Example:

    ```mlir
        bm.func @inference(%arg0: !bm.world<memref<*xf32, 7>>) attributes {llvm.emit_c_interface} {
            bm.print_world %arg0 : !bm.world<memref<*xf32, 7>>
            return
        }
    ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type
    );
    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >];
    let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    }];
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

def ReturnOp : BM_Op<"return", [NoSideEffect, HasParent<"FuncOp">,Terminator]> {
    let summary = "return operation";
    let description = [{
        The "return" operation represents a return operation within a function.
        The operation takes an optional tensor operand and produces no results.
        The operand type must match the signature of the function that contains
        the operation. For example:

        ```mlir
            bm.func @foo() -> tensor<2xf64> {
            ...
            bm.return %0 : tensor<2xf64>
            }
        ```
    }];

    // The return operation takes an optional input operand to return. This
    // value must match the return type of the enclosing function.
    let arguments = (ins Variadic<AnyType>:$input);

    // The return operation only emits the input in the format if it is present.
    let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

    // Allow building a ReturnOp with no return operand.
    let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
    ];

    // Provide extra utility definitions on the c++ operation class definition.
    let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
    }];

    // Indicate that additional verification for this operation is necessary.
    let hasVerifier = 1;
}
#endif // BM_OPS