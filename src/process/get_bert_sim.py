import sys

def get_index(path):
    fin = open(path, "r")
    line = fin.readline()
    index = []
    for line in fin:
        pieces = line.strip().split(",")
        index.append(pieces[0])
    fin.close()
    return index

def get_pairs(namepair, simfile):
    fin = open(namepair, "r")
    fin1 = open(simfile, "r")
    pairs = {}
    for line in fin:
        line1 = fin1.readline()
        pieces = line.strip().split("\t")
        pieces1 = line1.strip().split("\t")
        name1 = pieces[0]
        name2 = pieces[1]
        sim = pieces1[1]
        pairs[(name1, name2)] = sim
    fin.close()
    fin1.close()
    return pairs

def main():
    if len(sys.argv) != 5:
        print("using: %s indexfile namepair simfile outfile" % sys.argv[0])
        exit()
    indexfile = sys.argv[1]
    namepair = sys.argv[2]
    simfile = sys.argv[3]
    outfile = sys.argv[4]
    index = get_index(indexfile)
    pairs = get_pairs(namepair, simfile)
    n = len(index)
    simmatrix = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if (index[i], index[j]) in pairs:
                simmatrix[i][j] = pairs[(index[i], index[j])]
            else:
                print("not find", index[i], index[j])
    fout = open(outfile, "w")
    for xx in simmatrix:
        fout.write(",".join([str(x) for x in xx]))
        fout.write("\n")
    fout.close()

if __name__ == "__main__":
    main()

