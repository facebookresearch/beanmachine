from beanmachine.ppl.model.statistical_model import random_variable, functional

allowed_functions = {dict, list, set, super, random_variable, functional}

# TODO: Allowing these constructions raises additional problems that
# we have not yet solved. For example, what happens if someone
# searches a list for a value, but the list contains a graph node?
# And so on.
