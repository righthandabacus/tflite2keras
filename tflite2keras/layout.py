"""Layout transform helpers.

All weights are in numpy array, hence changing layout from NHWC to NCHW, for
example, can be easily done using np.moveaxis() or np.transpose(). Layout is
denoted as a string of distinct chars. Permutation pattern is resolved by
matching the chars between input layout and output layer.
"""

def getPerm(ilayout: str, olayout: str):
    char2index = {}
    for i in range(len(ilayout)):
        c = ilayout[i]
        char2index[c] = i

    perm = [char2index[c] for c in olayout]
    return perm


def transform(input, ilayout: str, olayout: str):
    if (ilayout == olayout):
        return input

    perm = getPerm(ilayout, olayout)
    transfrom_axis = [input[p] for p in perm]
    return transfrom_axis


class Layout(object):
    """Layout object that holds the intended layout transformation and the
    current layout pattern. To be associated with a tensor object"""
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target
        self.current = source

    def transform(self, input):
        output = transform(input, self.source, self.target)
        self.current = self.target
        return output

    @property
    def perm(self):
        return getPerm(self.source, self.target)

    def __str__(self):
        return self.current + '(' + self.source + '->' + self.target + ')'
