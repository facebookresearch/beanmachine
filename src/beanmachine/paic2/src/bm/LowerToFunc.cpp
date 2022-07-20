//
// Created by Steffi Stumpos on 7/20/22.
//
#include "mlir/IR/BuiltinDialect.h"
#include "bm/BMDialect.h"
#include "bm/passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include <iostream>

using namespace mlir;
//===----------------------------------------------------------------------===//
// BMToFunc RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<bm::ReturnOp> {
    using OpRewritePattern<bm::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(bm::ReturnOp op, PatternRewriter &rewriter) const final {
        // During this lowering, we expect that all function calls have been
        // inlined.
        if (op.hasOperand())
            return failure();

        // We lower "toy.return" directly to "func.return".
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
        return success();
    }
};
//===----------------------------------------------------------------------===//
// BMToFunc RewritePatterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<bm::PrintWorldOp> {
    using OpConversionPattern<bm::PrintWorldOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(bm::PrintWorldOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        // We don't lower "bm.print_world" in this pass, but we need to update its
        // operands.
        rewriter.updateRootInPlace(op,[&] { op->setOperands(adaptor.getOperands()); });
        return success();
    }
};
//===----------------------------------------------------------------------===//
// BMToFunc RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<bm::FuncOp> {
    using OpConversionPattern<bm::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(bm::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
        bool accepts_world_type = op.getFunctionType().getInput(0).getTypeID() == bm::WorldType().getTypeID();
        // we do not yet map return types
        if (op.getFunctionType().getNumResults() > 0 || !accepts_world_type || op.getFunctionType().getNumResults() > 1) {
            return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
                diag << "expected a bm function to accept only a world and return nothing";
            });
        }
        // TODO: iterate through parameters and update the type of the block arguments that are of type World
        // currently we expect a bm function to accept a world and we expect a world to be just
        // be a wrapper around a 1D Tensor. So let's map that to an unranked MemRef type
        mlir::Attribute memSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(rewriter.getContext(), 32), 7);
        //mlir::ShapedType unrankedTensorType = mlir::UnrankedMemRefType::get(rewriter.getF32Type(), memSpace);
        mlir::ShapedType rankedTensorType = mlir::MemRefType::get({5}, rewriter.getF64Type());
        llvm::SmallVector<mlir::Type> types;
        for(mlir::Type type_input : op.getFunctionType().getInputs()){
            if(type_input.dyn_cast_or_null<bm::WorldType>() != nullptr){
                types.push_back(rankedTensorType);
            } else {
                types.push_back(type_input);
            }
        }
        // TODO: needs to be recursive
        for(auto arg_ptr = op.front().args_begin(); arg_ptr != op.front().args_end(); arg_ptr++){
            mlir::BlockArgument blockArgument = *arg_ptr;
            if(blockArgument.getType().dyn_cast_or_null<bm::WorldType>() != nullptr){
                blockArgument.setType(rankedTensorType);
            }
        }
        mlir::TypeRange typeRange(types);
        // TODO: handle return value
        mlir::FunctionType new_function_type = rewriter.getFunctionType(typeRange, {});

        // Create a new function with an updated signature.
        auto newFuncOp = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName().str(), new_function_type);
        newFuncOp->setAttrs(op->getAttrs());
        newFuncOp.setType(new_function_type);
        rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),newFuncOp.end());


        // TODO: use the type converter
        /*
        TypeConverter typeConverter;
        llvm::SmallVector<Type> sv;
        sv.push_back(unrankedTensorType);
        typeConverter.convertTypes(bm::WorldType::get({unrankedTensorType}), sv);
        rewriter.convertRegionTypes(op.getCallableRegion(), typeConverter);
        op.dump();
         */

        rewriter.eraseOp(op);

        return success();
    }
};

//===----------------------------------------------------------------------===//
// BMToFuncLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
    struct BMToFuncLoweringPass
            : public PassWrapper<BMToFuncLoweringPass, OperationPass<ModuleOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BMToFuncLoweringPass)

        void getDependentDialects(DialectRegistry &registry) const override {
            registry.insert<func::FuncDialect, memref::MemRefDialect>();
        }
        void runOnOperation() final;
    };
} // namespace

void BMToFuncLoweringPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithmeticDialect,func::FuncDialect, memref::MemRefDialect>();
    target.addIllegalDialect<bm::BMDialect>();
    // PrintWorld will be handled in a separate lowering but we need to lower its operands here
    target.addDynamicallyLegalOp<bm::PrintWorldOp>([](bm::PrintWorldOp op) {
        return llvm::none_of(op->getOperandTypes(),
                             [](Type type) {
                                 bool is_world = type.isa<bm::WorldType>();
                                 return is_world;
                             });
    });

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Toy operations.
    RewritePatternSet patterns(&getContext());
    patterns.add<FuncOpLowering, PrintOpLowering, ReturnOpLowering>(&getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}



/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::bm::createLowerToFuncPass() {
    return std::make_unique<BMToFuncLoweringPass>();
}