import math

count = 0

def minimax(depth, index, max_turn, scores, target):
    global count
    if depth == target:
        count += 1
        return scores[index]
    if max_turn:
        return max(
            minimax(depth + 1, index * 2, False, scores, target),
            minimax(depth + 1, index * 2 + 1, False, scores, target),
        )
    return min(
        minimax(depth + 1, index * 2, True, scores, target),
        minimax(depth + 1, index * 2 + 1, True, scores, target),
    )

def alphabeta(depth, index, max_turn, scores, target, alpha, beta):
    global count
    if depth == target:
        count += 1
        return scores[index]
    if max_turn:
        value = float("-inf")
        for i in range(2):
            value = max(value, alphabeta(depth + 1, index * 2 + i, False, scores, target, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    value = float("inf")
    for i in range(2):
        value = min(value, alphabeta(depth + 1, index * 2 + i, True, scores, target, alpha, beta))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value

def prepare(scores):
    scores = list(scores)
    depth = math.ceil(math.log2(len(scores)))
    scores += [0] * (2 ** depth - len(scores))
    return scores, depth

def run(name, scores):
    global count
    scores, depth = prepare(scores)
    count = 0
    mm_value = minimax(0, 0, True, scores, depth)
    mm_leaves = count
    count = 0
    ab_value = alphabeta(0, 0, True, scores, depth, float("-inf"), float("inf"))
    ab_leaves = count
    count = 0
    left = minimax(1, 0, False, scores, depth) if depth > 0 else None
    right = minimax(1, 1, False, scores, depth) if depth > 0 else None
    if depth > 0:
        move = "Left" if left >= right else "Right"
    else:
        move = "No move (single leaf)"
    print(name)
    print("Leaves:", scores)
    print("Minimax value:", mm_value, "| Leaves evaluated:", mm_leaves)
    print("Alpha-Beta value:", ab_value, "| Leaves evaluated:", ab_leaves)
    print("Best move at root:", move)
    print("Values same:", mm_value == ab_value)
    print()

run("Test 1: Manual tree", [3, 5, 2, 9, 0, 7])
run("Test 2: Lab alpha-beta tree", [3, 5, 2, 9, 3, 5, 2, 9])
run("Test 3: Manual tree, reordered branches", [5, 3, 9, 2, 0, 0, 0, 7])
run("Edge case: all leaves equal", [4, 4, 4, 4, 4, 4, 4, 4])
