import numpy as np

def solve(pieces: list):
    total_difference = 0
    ordered_pieces = []
    for i in range(len(pieces)):
        pcopy = pieces.copy()
        temp_order = []
        anchor = pcopy[i]
        pcopy.pop(i)
        temp_order.append(anchor)




    return ordered_pieces


def return_edge(piece, edge):
    if edge == "t":
        return piece[0, :, :]
    elif edge == "b":
        return piece[-1, :, :]
    elif edge == "r":
        return piece[:, -1, :]
    elif edge == "l":
        return piece[:, 0, :]
    return None
