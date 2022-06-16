def _unindent(lines):
    # TODO: Handle the situation if there are tabs
    if len(lines) == 0:
        return lines
    num_spaces = len(lines[0]) - len(lines[0].lstrip(" "))
    if num_spaces == 0:
        return lines
    spaces = lines[0][0:num_spaces]
    return [(line[num_spaces:] if line.startswith(spaces) else line) for line in lines]