//
// Created by Steffi Stumpos on 7/20/22.
//
#include "bm/BMDialect.h"
#include "bm/passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// BMToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {
/// Lowers `bm.print_world` to a loop nest calling `printf` on each of the individual
/// elements of the array.
    class PrintWorldOpLowering : public ConversionPattern {
    public:
        explicit PrintWorldOpLowering(MLIRContext *context)
                : ConversionPattern(bm::PrintWorldOp::getOperationName(), 1, context) {}

        LogicalResult
        matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            auto memRefType = (*op->operand_type_begin()).cast<MemRefType>();
            auto memRefShape = memRefType.getShape();
            auto loc = op->getLoc();

            ModuleOp parentModule = op->getParentOfType<ModuleOp>();

            // Get a symbol reference to the printf function, inserting it if necessary.
            auto printfRef = getOrInsertPrintf(rewriter, parentModule);
            Value formatSpecifierCst = getOrCreateGlobalString(loc, rewriter, "frmt_spec", StringRef("%F \0", 4), parentModule);
            Value newLineCst = getOrCreateGlobalString(loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

            // Create a loop for each of the dimensions within the shape.
            SmallVector<Value, 4> loopIvs;
            for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
                auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
                auto upperBound =
                        rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
                auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
                auto loop =
                        rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
                for (Operation &nested : *loop.getBody())
                    rewriter.eraseOp(&nested);
                loopIvs.push_back(loop.getInductionVar());

                // Terminate the loop body.
                rewriter.setInsertionPointToEnd(loop.getBody());

                // Insert a newline after each of the inner dimensions of the shape.
                if (i != e - 1)
                    rewriter.create<func::CallOp>(loc, printfRef,
                                                  rewriter.getIntegerType(32), newLineCst);
                rewriter.create<scf::YieldOp>(loc);
                rewriter.setInsertionPointToStart(loop.getBody());
            }

            // Generate a call to printf for the current element of the loop.
            auto printOp = cast<bm::PrintWorldOp>(op);
            mlir::Value elementLoad = rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs);
           // mlir::Value constant = rewriter.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat((double)4.7), rewriter.getF64Type());
            // printf is a vararg function that takes at least one pointer to i8 (char in C), which returns an integer
            TypeRange results(rewriter.getIntegerType(32));
            ArrayRef<Value> print_operands({formatSpecifierCst, elementLoad});
            rewriter.create<func::CallOp>(loc, printfRef, results, print_operands);

            // Notify the rewriter that this operation has been removed.
            rewriter.eraseOp(op);
            return success();
        }

    private:
        /// Return a symbol reference to the printf function, inserting it into the
        /// module if necessary.
        static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                                   ModuleOp module) {
            auto *context = module.getContext();
            if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
                return SymbolRefAttr::get(context, "printf");

            // Create a function declaration for printf, the signature is:
            //   * `i32 (i8*, ...)`
            auto llvmI32Ty = IntegerType::get(context, 32);
            auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
            auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                    /*isVarArg=*/true);

            // Insert the printf function into the body of the parent module.
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
            return SymbolRefAttr::get(context, "printf");
        }

        /// Return a value representing an access into a global string with the given
        /// name, creating the string if necessary.
        static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                             StringRef name, StringRef value,
                                             ModuleOp module) {
            // Create the global at the entry of the module.
            LLVM::GlobalOp global;
            if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
                OpBuilder::InsertionGuard insertGuard(builder);
                builder.setInsertionPointToStart(module.getBody());
                auto type = LLVM::LLVMArrayType::get(
                        IntegerType::get(builder.getContext(), 8), value.size());
                global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                                        LLVM::Linkage::Internal, name,
                                                        builder.getStringAttr(value),
                        /*alignment=*/0);
            }

            // Get the pointer to the first character in the global string.
            Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
            Value cst0 = builder.create<LLVM::ConstantOp>(
                    loc, IntegerType::get(builder.getContext(), 64),
                    builder.getIntegerAttr(builder.getIndexType(), 0));
            return builder.create<LLVM::GEPOp>(
                    loc,
                    LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
                    globalPtr, ArrayRef<Value>({cst0, cst0}));
        }
    };
} // namespace

//===----------------------------------------------------------------------===//
// BMToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
    struct BMToLLVMLoweringPass
            : public PassWrapper<BMToLLVMLoweringPass, OperationPass<ModuleOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BMToLLVMLoweringPass)

        void getDependentDialects(DialectRegistry &registry) const override {
            registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
        }
        void runOnOperation() final;
    };
} // namespace

void BMToLLVMLoweringPass::runOnOperation() {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateAffineToStdConversionPatterns(patterns);
    mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);


    patterns.add<PrintWorldOpLowering>(&getContext());

    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}


std::unique_ptr<mlir::Pass> mlir::bm::createLowerToLLVMPass() {
    return std::make_unique<BMToLLVMLoweringPass>();
}