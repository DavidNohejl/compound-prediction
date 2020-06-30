import re
# pip install molmass
from molmass import Formula

class CompoundComposition:
    def __init__(self,H,C,N,O,P,S,Chl):
        self.H = H
        self.C = C
        self.N = N
        self.O = O
        self.P = P
        self.S = S
        self.Chl = Chl

    def features(self):
        return [self.H,self.C,self.N,self.N,self.P,self.S,self.Chl]

    def __str__(self):
          return "H:"+str(self.H)+ " C:"+str(self.C)+" N:"+str(self.N)+" O:"+str(self.O)+" P:"+str(self.P)+" S:"+str(self.S)+" Chl:"+str(self.Chl)

def compoundToComposition(compound, table):
    # https://stackoverflow.com/questions/2974362/parsing-a-chemical-formula
    element_pat = re.compile("([A-Z][a-z]?)(\d*)")
      
    h = c = n = o =p =s = chl = 0.0
    for (element_name, count) in element_pat.findall(compound):

        table.setdefault(element_name, 0)
        table[element_name] = table[element_name]+1
        if count == "":
            count = 1
        count = int(count)
        if element_name == 'H':
            h = count
        if element_name == 'C':
            c = count
        if element_name == 'N':
            n = count
        if element_name == 'O':
            o = count
        if element_name == 'P':
            p = count
        if element_name == 'S':
            s = count
        if element_name == 'Chl':
            chl = count

    return CompoundComposition(h,c,n,o,p,s,chl)
