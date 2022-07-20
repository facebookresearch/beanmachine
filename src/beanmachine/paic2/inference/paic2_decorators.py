import ast
import inspect
import paic2
import typing
from beanmachine.paic2.inference.utils import get_globals
from beanmachine.paic2.inference.to_paic2_ast import paic2_ast_generator
from beanmachine.ppl.world import World

def import_inference(entry_callable: typing.Callable):
    def wrapper(*args, **kwargs):
        inference_fnc = kwargs['inference']
        queries = kwargs['queries']
        observations = kwargs['observations']

        # configure world
        python_world = World.initialize_world(queries, observations)
        world_size = python_world.__len__()
        world_metadata = paic2.WorldSpec()
        world_metadata.set_print_name("print")
        world_metadata.set_world_name("MetaWorld")
        world_metadata.set_world_size(world_size)

        # compile inference
        lines, _ = inspect.getsourcelines(inference_fnc)
        source = "".join(lines)
        module = ast.parse(source)
        funcdef = module.body[0]
        to_paic = paic2_ast_generator()
        globals = get_globals(callable)
        python_function = to_paic.python_ast_to_paic_ast(funcdef, globals)

        # lower and execute inference
        mb = paic2.MLIRBuilder()
        mb.infer(python_function, world_metadata)
    return wrapper


def to_hardware(callable: typing.Callable):
    def wrapper(*args, **kwargs):
        mb = paic2.MLIRBuilder()
        lines, _ = inspect.getsourcelines(callable)
        source = "".join(lines)
        module = ast.parse(source)
        funcdef = module.body[0]
        to_paic = paic2_ast_generator()
        globals = get_globals(callable)
        python_function = to_paic.python_ast_to_paic_ast(funcdef, globals)
        arg = float(args[0])
        result = mb.evaluate(python_function, arg)
        return result
    return wrapper