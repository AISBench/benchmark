"""Judges for LiveCodeBench tasks with invalid tests or non-unique answers."""

import json
from fractions import Fraction
from math import isqrt


def prepare_test_cases(question_id, evaluation_sample):
    """Exclude inputs that violate the published constraints of abc392_f."""
    if question_id != 'abc392_f':
        return evaluation_sample, []

    cases = json.loads(evaluation_sample)
    if len(cases['inputs']) != len(cases['outputs']):
        raise ValueError('abc392_f input/output case counts differ')

    kept_inputs, kept_outputs, excluded = [], [], []
    for index, (raw_input, expected) in enumerate(
            zip(cases['inputs'], cases['outputs'])):
        try:
            values = [int(token) for token in raw_input.split()]
            n = values[0]
            valid = (1 <= n <= 500000 and len(values) == n + 1
                     and all(1 <= p <= i
                             for i, p in enumerate(values[1:], 1)))
        except (ValueError, IndexError, AttributeError):
            valid = False
        if valid:
            kept_inputs.append(raw_input)
            kept_outputs.append(expected)
        else:
            excluded.append(index)

    if not kept_inputs:
        raise ValueError('abc392_f has no valid test cases after filtering')
    cases['inputs'], cases['outputs'] = kept_inputs, kept_outputs
    return json.dumps(cases), excluded


def _integer_cuberoot(n):
    lo, hi = 0, 1
    while hi * hi * hi <= n:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid * mid * mid <= n:
            lo = mid
        else:
            hi = mid
    return lo


def _positive_cube_difference_exists(n):
    if n <= 0:
        return False
    for difference in range(1, _integer_cuberoot(n - 1) + 1):
        if n % difference:
            continue
        numerator = 4 * (n // difference) - difference * difference
        if numerator <= 0 or numerator % 3:
            continue
        square = numerator // 3
        root = isqrt(square)
        if root * root == square and root > difference \
                and (root - difference) % 2 == 0:
            return True
    return False


def _judge_abc397_d(raw_input, raw_output):
    try:
        n = int(raw_input.strip())
        tokens = raw_output.split()
        if tokens == ['-1']:
            return not _positive_cube_difference_exists(n)
        if len(tokens) != 2:
            return False
        x, y = map(int, tokens)
        return x > 0 and y > 0 and x**3 - y**3 == n
    except (ValueError, TypeError):
        return False


def _prime_factors(n):
    factors = []
    divisor = 2
    while divisor * divisor <= n:
        if n % divisor == 0:
            factors.append(divisor)
            while n % divisor == 0:
                n //= divisor
        divisor += 1 if divisor == 2 else 2
    if n > 1:
        factors.append(n)
    return factors


def _judge_arc191_c(raw_input, raw_output):
    try:
        values = [int(token) for token in raw_input.split()]
        count = values[0]
        tokens = [int(token) for token in raw_output.split()]
        if len(values) != count + 1 or len(tokens) != 2 * count:
            return False
        for index, order in enumerate(values[1:]):
            a, modulus = tokens[2 * index:2 * index + 2]
            if not (1 <= a <= 10**18 and 1 <= modulus <= 10**18):
                return False
            if modulus == 1:
                if order != 1:
                    return False
                continue
            if pow(a, order, modulus) != 1:
                return False
            # An order dividing N equals N iff it does not divide N/p
            # for any distinct prime factor p of N.
            if any(pow(a, order // prime, modulus) == 1
                   for prime in _prime_factors(order)):
                return False
        return True
    except (ValueError, TypeError, IndexError):
        return False


def _judge_arc195_c(raw_input, raw_output):
    try:
        values = [int(token) for token in raw_input.split()]
        count = values[0]
        if len(values) != 1 + 2 * count:
            return False
        tokens = raw_output.split()
        cursor = 0
        for case in range(count):
            red, blue = values[1 + 2 * case:3 + 2 * case]
            if cursor >= len(tokens):
                return False
            # Red moves change chessboard colour; blue moves preserve it.
            possible = red % 2 == 0 and (red != 0 or blue % 2 == 0)
            answer = tokens[cursor].lower()
            cursor += 1
            if answer == 'no':
                if possible:
                    return False
                continue
            if answer != 'yes' or not possible:
                return False
            placements = []
            for _ in range(red + blue):
                if cursor + 3 > len(tokens):
                    return False
                color = tokens[cursor].upper()
                row, column = map(int, tokens[cursor + 1:cursor + 3])
                cursor += 3
                if color not in ('R', 'B') or not (
                        1 <= row <= 10**9 and 1 <= column <= 10**9):
                    return False
                placements.append((color, row, column))
            if sum(color == 'R' for color, _, _ in placements) != red:
                return False
            if len({(row, col) for _, row, col in placements}) != len(placements):
                return False
            for index, (color, row, column) in enumerate(placements):
                _, next_row, next_column = placements[(index + 1) % len(placements)]
                dr, dc = abs(row - next_row), abs(column - next_column)
                if color == 'R' and dr + dc != 1:
                    return False
                if color == 'B' and (dr, dc) != (1, 1):
                    return False
        return cursor == len(tokens)
    except (ValueError, TypeError, IndexError):
        return False


def judge_stdio(question_id, raw_input, raw_output):
    """Return None for ordinary tasks; otherwise validate by task rules."""
    if question_id == 'abc397_d':
        return _judge_abc397_d(raw_input, raw_output)
    if question_id == 'arc191_c':
        return _judge_arc191_c(raw_input, raw_output)
    if question_id == 'arc195_c':
        return _judge_arc195_c(raw_input, raw_output)
    return None


def _separate_squares_answer(squares):
    events = []
    total_area = 0
    for _, y, side in squares:
        events.append((y, side))
        events.append((y + side, -side))
        total_area += side * side
    events.sort()
    area_twice = 0
    slope = 0
    previous_y = events[0][0]
    index = 0
    while index < len(events):
        y = events[index][0]
        if area_twice == total_area:
            return Fraction(previous_y)
        next_area_twice = area_twice + 2 * slope * (y - previous_y)
        if slope > 0 and next_area_twice >= total_area:
            return Fraction(previous_y) + Fraction(
                total_area - area_twice, 2 * slope)
        area_twice = next_area_twice
        while index < len(events) and events[index][0] == y:
            slope += events[index][1]
            index += 1
        previous_y = y
    raise ValueError('Area bisector was not found')


def judge_call(question_id, inputs, output):
    if question_id != '3763':
        return None
    try:
        if len(inputs) != 1 or isinstance(output, bool):
            return False
        answer = Fraction(str(output))
        expected = _separate_squares_answer(inputs[0])
        return abs(answer - expected) <= Fraction(1, 100000)
    except (ValueError, TypeError, ZeroDivisionError, IndexError):
        return False
