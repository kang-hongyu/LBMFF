import csv
from rdkit import DataStructs
from rdkit import Chem
f=open('../data/SCMFDD-S/drug.csv','r')
f_csv=csv.reader(f)
smiles = []
druglist = []
for row in f_csv:
   smiles.append(row[1])
   druglist.append(row[0])
druglist = druglist[1:]
#print(smiles)

def cosine_similarity(vector1, vector2):
  dot_product = 0.0
  normA = 0.0
  normB = 0.0
  for a, b in zip(vector1, vector2):
    dot_product += a * b
    normA += a ** 2
    normB += b ** 2
  if normA == 0.0 or normB == 0.0:
    return 0
  else:
    return dot_product / ((normA**0.5)*(normB**0.5))

#计算smiles相似度
sims = []
smiles=smiles[1:]
for i in range(len(smiles)):
    sims.append([])
    m1 = Chem.MolFromSmiles(smiles[i], True)
    if m1 is None:
        m1 = Chem.MolFromSmiles(smiles[i], False)
    fps1 = Chem.RDKFingerprint(m1)
    #print("i:",i)
    for j in range(len(smiles)):
        #print("j:",j)
        m2 = Chem.MolFromSmiles(smiles[j], True)
        if m2 is None:
            m2 = Chem.MolFromSmiles(smiles[j], False)
        fps2 = Chem.RDKFingerprint(m2)
        s1 = DataStructs.FingerprintSimilarity(fps1,fps2)
        sims[-1].append(s1)
fsim = open("../data/SCMFDD-S/drugsim_smiles.csv", "w")
for fs in sims:
    fsim.write(",".join([str(x) for x in fs]))
    fsim.write("\n")
fsim.close()
#计算靶标相似度
f1 = open('../data/SCMFDD-S/drug-protein.csv','r')
f1_csv = csv.reader(f1)
drug_proteins = {}
proteins = set()
drug_protein_feature = []
drug_protein_sims = []
first = 0
for row in f1_csv:
    first += 1
    if first == 1:
        continue
    if row[0] not in drug_proteins:
        drug_proteins[row[0]] = {}
    drug_proteins[row[0]][row[1]] = 1
    proteins.add(row[1])
proteinslist = list(proteins)
for drug in druglist:
    if drug not in drug_proteins:
        drug_protein_feature.append([0 for i in range(len(proteinslist))])
        continue
    drug_protein_feature.append([])
    prots = drug_proteins[drug]
    for prot in proteinslist:
        if prot in prots:
            drug_protein_feature[-1].append(1)
        else:
            drug_protein_feature[-1].append(0)
for i,v1 in enumerate(drug_protein_feature):
    drug_protein_sims.append([])
    for j,v2 in enumerate(drug_protein_feature):
        if i == j:
            drug_protein_sims[-1].append(1.0)
        else:
            drug_protein_sims[-1].append(cosine_similarity(v1,v2))
fsim1 = open("../data/SCMFDD-S/drugsim_protein.csv", "w")
for fs in drug_protein_sims:
    fsim1.write(",".join([str(x) for x in fs]))
    fsim1.write("\n")
fsim1.close()
f1.close()

#计算副作用相似度
f2 = open('../data/SCMFDD-S/drug-side.csv','r')
f2_csv = csv.reader(f2)
drug_sides = {}
sides = set()
drug_side_feature = []
drug_side_sims = []
first = 0
for row in f2_csv:
    first += 1
    if first == 1:
        continue
    if row[0] not in drug_sides:
        drug_sides[row[0]] = {}
    drug_sides[row[0]][row[1]] = 1
    sides.add(row[1])
sideslist = list(sides)
for drug in druglist:
    if drug not in drug_sides:
        drug_side_feature.append([0 for i in range(len(sideslist))])
        continue
    drug_side_feature.append([])
    sids = drug_sides[drug]
    for sid in sideslist:
        if sid in sids:
            drug_side_feature[-1].append(1)
        else:
            drug_side_feature[-1].append(0)
for i,v1 in enumerate(drug_side_feature):
    drug_side_sims.append([])
    for j,v2 in enumerate(drug_side_feature):
        if i == j:
            drug_side_sims[-1].append(1.0)
        else:
            drug_side_sims[-1].append(cosine_similarity(v1,v2))
fsim2 = open("../data/SCMFDD-S/drugsim_side.csv", "w")
for fs in drug_side_sims:
    fsim2.write(",".join([str(x) for x in fs]))
    fsim2.write("\n")
fsim2.close()
f2.close()

#a = 0.3
#b = 0.3
#c = 0.4
for a in range(0,11):
    for b in range(0,11-a):
        c = 10 - a - b
        fname = "drugsim_%d_%d_%d.csv" % (a,b,c)
        drug_sims_all = [[0.0 for i in range(len(druglist))] for j in range(len(druglist))]
        for i in range(len(druglist)):
            for j in range(len(druglist)):
                drug_sims_all[i][j] = (a*sims[i][j] + b*drug_protein_sims[i][j] + c*drug_side_sims[i][j])/10
        fsim3 = open("../data/SCMFDD-S/drugsim/" + fname, "w")
        for fs in drug_sims_all:
            fsim3.write(",".join([str(x) for x in fs]))
            fsim3.write("\n")
        fsim3.close()

f.close()
