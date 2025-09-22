from random import randint
from matrix import Matrix
for n in range(1, 4):
    with open(f'goida11/{n}', 'w') as filename:
        N = randint(2, 5)
        a, b, g = randint(-10, 10), randint(-10, 10), randint(-10, 10)
        print(N,a,b,g, file=filename)
        A = Matrix([[randint(-10, 10) for i in range(N)] for j in range(N)])
        B = Matrix([[randint(-10, 10) for i in range(N)] for j in range(N)])
        C = Matrix([[randint(-10, 10) for i in range(N)] for j in range(N)])
        X = Matrix([[randint(-10, 10) for i in range(N)] for j in range(N)])
        print(A, file=filename)
        print(B, file=filename)
        print(C, file=filename)
        print(X, file=filename)
for n in range(4, 24):
    with open(f'goida11/{n}', 'w') as filename:
        N = randint(10, 20)
        a, b, g = randint(-100, 100), randint(-100, 100), randint(-10, 100)
        print(N,a,b,g, file=filename)
        A = Matrix([[randint(-100, 100) for i in range(N)] for j in range(N)])
        B = Matrix([[randint(-100, 100) for i in range(N)] for j in range(N)])
        C = Matrix([[randint(-100, 100) for i in range(N)] for j in range(N)])
        X = Matrix([[randint(-100, 100) for i in range(N)] for j in range(N)])
        print(A, file=filename)
        print(B, file=filename)
        print(C, file=filename)
        print(X, file=filename)
