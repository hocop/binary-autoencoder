
class GeneratorLen(object):
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length
    def __len__(self): 
        return self.length
    def __iter__(self):
        return self.gen

class GeneratorFile(object):
    def __init__(self, fname):
        self.length = 0
        for _ in open(fname):
            self.length += 1
        self.gen = open(fname)
    def __len__(self): 
        return self.length
    def __iter__(self):
        return self.gen
