# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict

import torch.tensor as tensor
from beanmachine.ppl.world.variable import Variable


class World(object):
    """
    Represents the world through inference run.

    takes in:
        init_world_likelihood: the likelihood of the initial world being passed
        in.
        init_world: the initial world from which the inference algorithm can
        start from. (helps us to support resumable inference)

    parameters are:
        variables_: a dict of variables keyed with their function signature.
        log_prob_: joint log probability of the world
        diff_: it is a subset of the variables_ thats includes changes to the
               world variables_.
        diff_log_update_: it is the variable that tracks the update in the diff
                          log probability.

    for instance for model below:


    @sample
    def foo():
        return dist.Bernoulli(torch.tensor(0.1))

    @sample
    def bar():
        if not foo().item():
            return dist.Bernoulli(torch.tensor(0.1))
        else:
            return dist.Bernoulli(torch.tensor(0.9))


    World.variables_ will be:

    defaultdict(<class 'beanmachine.ppl.utils.variable.Variable'>,
    {
     RandomVariable(function=<function bar at 0x7f6c82b0e488>, arguments=()):
        Variable(
                 distribution=Bernoulli(probs: 0.8999999761581421,
                                        logits: 2.1972243785858154),
                 value=tensor(0.),
                 parent={RandomVariable(function=<function foo at 0x7f6d343c8bf8>, arguments=())},
                 children=set(),
                 log_prob=tensor(-2.3026)
                ),
     RandomVariable(function=<function foo at 0x7f6d343c8bf8>, arguments=()):
         Variable(
                  distribution=Bernoulli(probs: 0.10000000149011612,
                                         logits: -2.1972246170043945),
                  value=tensor(0.),
                  parent=set(),
                  children={RandomVariable(function=<function bar at 0x7f6c82b0e488>, arguments=())},
                  log_prob=tensor(-0.1054)
                 )
     }
    )
    """

    def __init__(self, init_world_log_prob=None, init_world_dict=None):
        self.variables_ = defaultdict(Variable)
        self.diff_ = defaultdict(Variable)
        self.log_prob_ = tensor(0.0)
        self.diff_log_update_ = tensor(0.0)

    def add_node_to_world(self, node, var):
        """
        Add the node to the world. Since all updates are done through diff_, here
        we will just update diff_.
        """
        self.diff_[node] = var

    def update_diff_log_prob(self, node):
        """
        Adds the log update to diff_log_update_
        """
        self.diff_log_update_ += self.diff_[node].log_prob - (
            self.variables_[node].log_prob if node in self.variables_ else tensor(0.0)
        )

    def get_node_in_world(self, node):
        """
        Get the node in the world, by first looking up diff_, if not available there
        and it can find it in variables, it creates a copy of variables_ node into
        diff_ and returns the new diff_ node and if not available in any, return
        None

        parameters are:
            node: node to be looked up in world

        returns:
            the corresponding node from the world
        """
        if node in self.diff_:
            return self.diff_[node]
        elif node in self.variables_:
            self.diff_[node] = self.variables_[node].copy()
            return self.diff_[node]
        return None

    def contains_in_world(self, node):
        """
        Looks up both variables_ and diff_ and returns true if node is available
        in any of them, otherwise, returns false

        parameters are:
            node: node to be looked up in the world

        returns:
            true if found else false
        """
        if node in self.diff_ or node in self.variables_:
            return True
        return False

    def get_all_world_vars(self):
        """
        Returns all variables in the world
        """
        return self.variables_

    def update_world(self):
        """
        If changes in a diff is accepted, world's variables_ are updated with
        their corrseponding diff_ value.
        """
        for node in self.diff_:
            var = self.diff_[node]
            self.variables_[node] = var

        self.log_prob_ += self.diff_log_update_
        self.diff_ = defaultdict(Variable)
        self.diff_log_update_ = tensor(0.0)

    def start_diff_with_proposed_val(self, node, proposed_value):
        """
        Starts a diff with new value for node.

        parameters are:
            node: the node who has a new proposed value
            proposed_value: the proposed value for node

        returns:
            difference of old and new log probability of the node after updating
            the node value to the proposed value
        """
        self.diff_ = defaultdict(Variable)
        var = self.variables_[node].copy()
        old_log_prob = var.log_prob
        var.value = proposed_value
        var.log_prob = var.distribution.log_prob(proposed_value)
        self.diff_[node] = var
        node_log_update = var.log_prob - old_log_prob
        self.diff_log_update_ += node_log_update
        return node_log_update

    def create_child_with_new_distributions(self, node, stack):
        """
        Adds all node's children to diff_ and re-computes their distrbutions
        and log_prob

        parameters are:
            node: the node whose value was just updated to a proposed value and
            thus its children's distributions are needed to be recomputed.
            model: the statistical model

        returns:
            difference of old and new log probability of the immediate children
            after updating the node value to the proposed value
        """
        old_log_probs = tensor(0.0)
        new_log_probs = tensor(0.0)
        for child in self.diff_[node].children.copy():
            child_func, child_args = child
            child_var = self.variables_[child].copy()
            old_log_probs += child_var.log_prob
            child_var.parent = set()
            self.diff_[child] = child_var
            stack.append(child)
            child_var.distribution = child_func(*child_args)
            stack.pop()
            child_var.log_prob = child_var.distribution.log_prob(child_var.value).sum()
            new_log_probs += child_var.log_prob

        children_log_update = new_log_probs - old_log_probs
        self.diff_log_update_ += children_log_update
        return children_log_update

    def propose_change(self, node, proposed_value, stack):
        """
        Creates the diff for the proposed change

        parameters are:
            node: the node who has a new proposed value
            proposed_value: the proposed value for node
            propose_change: difference in old and newly proposed value log
            model: statistical  model

        returns:
            difference of old and new log probability of node's children
            difference of old and new log probability of world
        """
        node_log_update = self.start_diff_with_proposed_val(node, proposed_value)
        children_node_log_update = self.create_child_with_new_distributions(node, stack)
        world_log_update = self.diff_log_update_
        return children_node_log_update, world_log_update, node_log_update
