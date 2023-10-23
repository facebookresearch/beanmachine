# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Any, Dict, List, Set, Tuple

from beanmachine.facebook.goal_inference.doors_keys_gems.dkg_definitions import (
    Direction,
    DKGState,
    Gem,
    Item,
    Key,
)


def parse(file_path: str) -> DKGState:
    """Reads in selected problem. Intializes state by parsing file for objects, conditions, and goal.

    Arguments:
        file_path: path to the selected problem

    Returns:
        dkg_state: Initial state of the doors, keys, and gems problem
    """
    full_file = open(file_path, "r")
    full_problem = full_file.read()
    full_file.close()

    data_parse = full_problem.split(":")
    objs_data = ""
    condition_data = ""
    goal_data = ""

    for data_type in data_parse:
        if data_type[:7] == "objects":
            objs_data = data_type

        elif data_type[:4] == "init":
            condition_data = data_type

        elif data_type[:4] == "goal":
            goal_data = data_type

    items, gems, keys, directions = parse_objects(objs_data)

    width, height, xdiff, ydiff, doors, walls, x, y, at = parse_conditions(
        condition_data
    )

    goal = parse_goal(goal_data)

    has = {}

    return DKGState(
        goal,
        width,
        height,
        xdiff,
        ydiff,
        doors,
        walls,
        items,
        directions,
        keys,
        gems,
        has,
        at,
        x,
        y,
    )


def parse_objects(
    obj_string: str,
) -> Tuple[Dict[str, Item], Dict[str, Gem], Dict[str, Key], Dict[str, Direction]]:
    """Sets State Objects [items,gems,keys] by parsing object substring of PDDL file.

    Arguments:
        obj_string: The substring of the PDDL file where objects are defined

    Returns:
        items: A map from Item names to the Item objects
        gems: A map from Gem names to the Gem objects
        keys: A map from Key names to the Key objects
        directions: A map from Direction names to the Direction objects
    """
    items = {}
    gems = {}
    keys = {}
    directions = {}

    obj_dict = obj_parser(obj_string)
    if "gem" in obj_dict:
        for g in obj_dict["gem"]:
            new_gem = Gem(g)
            gems[g] = new_gem
            items[g] = new_gem

    if "key" in obj_dict:
        for k in obj_dict["key"]:
            new_key = Key(k)
            keys[k] = new_key
            items[k] = new_key

    if "direction" in obj_dict:
        for d in obj_dict["direction"]:
            new_direction = Direction(d)
            directions[d] = new_direction

    return items, gems, keys, directions


def parse_param_condition(
    condition: List[str],
) -> Tuple[Any, ...]:
    """Handles condition parsing for setting of environment parameters

    Arguments:
        condition: A string in the form e.g. (= width 3) that relates to a parameter of the environment

    Returns:
        parsed_condition: The condition parsed into a more easily manipulated format (variable, *args)
    """
    # Parameters can appear with and without paranthesis
    if condition[1][0] == "(":
        condition[1] = condition[1][1:]
    if condition[1][-1] == ")":
        condition[1] = condition[1][:-1]

    if condition[1] == "xpos":
        return ("x", int(condition[2]))
    elif condition[1] == "ypos":
        return ("y", int(condition[2]))
    elif condition[1] == "width":
        return ("width", int(condition[2]))
    elif condition[1] == "height":
        return ("height", int(condition[2]))
    elif condition[1] == "xdiff":
        return ("xdiff", condition[2][:-1], int(condition[-1]))
    elif condition[1] == "ydiff":
        return ("ydiff", condition[2][:-1], int(condition[-1]))
    else:
        raise ValueError("Unknown Parameter")


def parse_conditions(
    condition_string: str,
) -> Tuple[
    int,
    int,
    Dict[str, int],
    Dict[str, int],
    Set[Tuple[int, int]],
    Set[Tuple[int, int]],
    int,
    int,
    Dict[str, Tuple[int, int]],
]:
    """Sets parameters [x,y,width,height,xdiff,ydiff] and locations of doors/walls/items

    Arguments:
        condition_string: The substring of the PDDL file where conditions are defined

    Returns:
        width: Width of environment
        height: Height of environment
        xdiff: Movement/Interaction along x-axis when given a direction
        ydiff: Movement/Interaction along y-axis when given a direction
        doors: Positions of doors
        walls: Positions of walls
        x: X-position of Agent
        y: Y-position of Agent
        at: Map from name to location for all Items not held by the Agent

    """
    params = {}
    params["xdiff"] = {}
    params["ydiff"] = {}
    doors = set()
    walls = set()
    at = {}

    # Separate conditions
    conditions = condition_parser(condition_string)
    for condition in conditions:
        condition_splt = condition.split()
        # Identify if condition relates to an environment variable
        if condition_splt[0] == "=":
            condition_parsed = parse_param_condition(condition_splt)
            if len(condition_parsed) == 3:
                params[condition_parsed[0]][condition_parsed[1]] = condition_parsed[2]
            else:
                params[condition_parsed[0]] = condition_parsed[1]

        elif condition_splt[0] == "wall":
            walls.add((int(condition_splt[1]), int(condition_splt[2])))

        elif condition_splt[0] == "at":
            at[condition_splt[1]] = (int(condition_splt[2]), int(condition_splt[3]))

        elif condition_splt[0] == "doorloc":
            doors.add((int(condition_splt[1]), int(condition_splt[2])))

        elif condition_splt[0] == "itemloc" or condition_splt[0] == "door":
            pass

        else:
            raise ValueError("Unknown Rule")

    return (
        params["width"],
        params["height"],
        params["xdiff"],
        params["ydiff"],
        doors,
        walls,
        params["x"],
        params["y"],
        at,
    )


def parse_goal(goal_string: str) -> Tuple[str, str]:
    """Parses goal into predicate and argument
    e.g. "has gem1" -> ("has","gem1")

    Arguments:
        goal_string: Substring of PDDL file related to the goal. Example: "goal (has gem3)"

    Returns:
        new_goal: Goal in the format ("has","gem1")

    """
    # Clean text around goal
    goal = goal_parser(goal_string)
    splt = goal.split()
    new_goal = (splt[0], splt[1:][0])
    return new_goal


def obj_parser(obj_string: str) -> Dict[str, List]:
    """Parses Objects listed in PDDL file into Dictionary e.g obj_dict["key"] = ["key1","key2","key3"...]"""
    # Cleaning Input
    obj_string = obj_string[7:]
    obj_string = obj_string.replace("\n", "")
    obj_string = obj_string.replace(")", "")
    obj_string = obj_string.replace("(", "")
    obj_split = obj_string.split()

    # Create Dict
    obj_dict = {}

    active_names = []
    index = 0
    while index < len(obj_split):
        if obj_split[index] == "-":
            index += 1
            obj_type = obj_split[index]
            obj_dict[obj_type] = active_names
            active_names = []

        else:
            active_names.append(obj_split[index])

        index += 1

    return obj_dict


def condition_parser(condition_string: str) -> List[str]:
    """Parses text stream of initial conditions into list of single conditions
    e.g. "(wall 2 7)
         (wall 4 7)
         (at key1 5 7)
         (itemloc 5 7)"

    -> ["wall 2 7","wall 4 7","at key1 5 7","itemloc 5 7"]

    Arguments:
        condition_string: Substring of PDDL file related to starting conditions

    Returns:
        condition_split: List of separate conditions
    """

    # Cleaning Object
    condition_string = condition_string[4:]
    condition_string = condition_string.replace("\n", "")

    # Get separate rules
    condition_string = " ".join(condition_string.split())
    condition_split = condition_string.split(") (")

    # Fix off by one
    condition_split = condition_split[:-1]
    condition_split[0] = condition_split[0][1:]
    last_pos = len(condition_split[-1]) - 1
    while not condition_split[-1][last_pos].isalnum():
        last_pos -= 1
    condition_split[-1] = condition_split[-1][: last_pos + 1]

    return condition_split


def goal_parser(goal_string: str) -> str:
    """Cleans text around goal phrase e.g. "goal (has gem3)" -> "has gem3"

    Arguments:
        goal_string: goal in format "goal (has gem3)"
    Returns:
        goal: goal in format "has gem3"

    """
    return goal_string[goal_string.find("(") + 1 : goal_string.find(")")]
