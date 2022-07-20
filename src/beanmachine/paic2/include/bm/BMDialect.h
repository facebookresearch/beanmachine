//
// Created by Steffi Stumpos on 7/20/22.
//

#ifndef PAIC2_BMDIALECT_H
#define PAIC2_BMDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// This class defines the struct type. It represents a collection
// of function pointers and an array.
namespace mlir {
    namespace bm {
        struct WorldTypeStorage;
    }
}

#include "bm/bm_dialect.h.inc"
#define GET_OP_CLASSES
#include "bm/bm_ops.h.inc"

namespace mlir {
    namespace bm {
        class WorldType : public mlir::Type::TypeBase<WorldType, mlir::Type,
                WorldTypeStorage> {
        public:
            /// Inherit some necessary constructors from 'TypeBase'.
            using Base::Base;
            static WorldType get(llvm::ArrayRef<mlir::Type> elementTypes);
            llvm::ArrayRef<mlir::Type> getElementTypes();
            size_t getNumElementTypes() { return getElementTypes().size(); }
        };
    }
}

#endif //PAIC2_BMDIALECT_H
