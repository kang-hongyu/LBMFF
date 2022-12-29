def get_drug_index(path):
    fin = open(path, "r")
    line = fin.readline()
    drugindex = {}
    druglist = []
    idx = 0
    for line in fin:
        pieces = line.split(",")
        drug_id = pieces[0]
        druglist.append(drug_id)
        drugindex[drug_id] = idx
        idx += 1
    fin.close()
    return drugindex,druglist

def get_disease_index(path):
    fin = open(path, "r")
    line = fin.readline()
    diseaseindex = {}
    diseaselist = []
    idx = 0
    for line in fin:
        pieces = line.split(",")
        disease_id = pieces[0]
        diseaselist.append(disease_id)
        diseaseindex[disease_id] = idx
        idx += 1
    fin.close()
    return diseaseindex,diseaselist

def get_drug_disease_pair(path):
    fin = open(path, "r")
    line = fin.readline()
    drug_disease_pair = {}
    for line in fin:
        pieces = line.strip().split(",")
        drug_id = pieces[0]
        disease_id = pieces[1]
        drug_disease_pair[(drug_id, disease_id)] = 1
    return drug_disease_pair

def main():
    data_path = "../data/SCMFDD-S/"
    drugindex,druglist = get_drug_index(data_path + "drug.csv")
    diseaseindex,diseaselist = get_disease_index(data_path + "disease.csv")
    drug_disease_pair = get_drug_disease_pair(data_path + "drug_disease.csv")
    predict_file = data_path + "predict_0.csv"
    predict_pair = []
    threshold = 0.7
    topk = 1000000
    ndrug = len(drugindex)
    ndisease = len(diseaseindex)
    fpredict = open(predict_file, "r")
    idx = 0
    for line in fpredict:
        line = line.strip()
        pieces = line.split(",")
        drug = druglist[idx]
        for i in range(len(pieces)):
            if float(pieces[i]) > threshold and (drug, diseaselist[i]) not in drug_disease_pair:
                predict_pair.append((drug, diseaselist[i], pieces[i]))
        idx += 1
    predict_pair = sorted(predict_pair, key = lambda x:x[2], reverse = True)
    fout = open(data_path + "predict_result.csv", "w")
    total = len(predict_pair)
    for i in range(topk):
        if i >= total:
            break
        fout.write(",".join([str(x) for x in predict_pair[i]]))
        fout.write("\n")
    fout.close()
    fpredict.close()

if __name__ == "__main__":
    main()

