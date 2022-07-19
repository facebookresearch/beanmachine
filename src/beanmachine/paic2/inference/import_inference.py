import ast
import inspect
import paic_mlir
import typing

import beanmachine.ppl.compiler.bm_to_bmg
import beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.to_paic_ast
import beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.utils

def import_inference(entry_callable: typing.Callable):
    def wrapper(*args, **kwargs):
        # step 1: WORLD METADATA COLLECTION
        # get world directions by invoking inference_fnc with a world tracer.
        # note that to start I'm skipping this step and providing a reasonable standin
        inference_fnc = kwargs['inference']
        functions_to_generate = paic_mlir.LogProbQueryList()
        functions_to_generate.push_back(paic_mlir.LogProbQueryTypes.TARGET)
        world_metadata = paic_mlir.WorldClassSpec(functions_to_generate)
        world_metadata.set_print_name("print")
        world_metadata.set_world_name("MetaWorld")
        mb = paic_mlir.MLIRBuilder()

        # step 2: INFERENCE COMPILATION
        # compile the inference_fnc, using the world directions to modify the world calls to match the interface
        # identified in step 1. At the end of this step, we should have a function to create a world and a function that accepts a world
        # TODO: include an ast created from "paic_ast_generator" with python_function
        lines, _ = inspect.getsourcelines(inference_fnc)
        source = "".join(beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.utils._unindent(lines))
        module = ast.parse(source)
        funcdef = module.body[0]
        to_paic = beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.to_paic_ast.paic_ast_generator()
        globals = beanmachine.ppl.compiler.bm_to_bmg._get_globals(inference_fnc)
        python_function = to_paic.python_ast_to_paic_ast(funcdef, globals)
        inference_functions = mb.create_inference_functions(python_function, world_metadata)

        # step 3:  MODEL COMPILATION
        # create a world from the model by tracing the queries and observations to create a graph and ungraphing according to trace
        variables = paic_mlir.Tensor()
        for i in range(0, 5):
            variables.push_back(float(i))
        world = inference_functions.world_with_variables(variables)

        # step 4: invoke the compiled inference function with the world you just created
        inference_functions.inference_function(world)
    return wrapper