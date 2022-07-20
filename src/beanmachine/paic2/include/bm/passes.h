//
// Created by Steffi Stumpos on 7/20/22.
//

#ifndef PAIC2_PASSES_H
#define PAIC2_PASSES_H
#include <memory>

namespace mlir {
    class Pass;

    namespace bm {
        std::unique_ptr<mlir::Pass> createLowerToFuncPass();
        std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
    }
}

#endif //PAIC2_PASSES_H
