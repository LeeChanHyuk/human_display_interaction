class dada():
    a = 3
    b = 4
    def __init__(self, c, d) -> None:
        self.a = c
        self.b = d
        pass

dada_instance = dada(1, 2)
print(dada_instance.a, dada_instance.b)