import itertools
import math

def prime_provid(n):
    """n = last_number"""
    numbers = set(range(n, 1, -1))
    primes = []
    while numbers:
        x = numbers.pop()
        primes.append(x)
        numbers -= set(range(2 * x, n + 1, x))
    return primes


def prime_factors(n):
    """find prime factors of a number, return dictionary of numbers with the times one of them repeated"""
    primes = {}
    while n % 2 == 0:
        primes[f"{2}"] = primes.get(f"{2}", 0) + 1
        n = n // 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            primes[f"{i}"] = primes.get(f"{i}", 0) + 1
            n = n // i
    if n > 2:
        primes[f"{n}"] = 1
    return primes


def factors(n: int) -> list:
    """returns all the factors for number 'n'"""
    table = []
    for i in range(1, n):
        if divmod(n, i)[1] == 0:
            table.append(i)
    return table


def findpermutations(l: list) -> list:
    """it will returns all permutations off a list as a list"""
    return list(itertools.permutations(l))
