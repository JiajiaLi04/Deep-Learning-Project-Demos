def add(a, b, c=None):
    p = a + b

    if c is not None:
        c = c
    else:
        c = 1
    e = c + 1
    return e


print(add(1, 2, 2))
a = add(1, 2)
print(a)
