import sys

def get_sim(filepath):
  fin = open(filepath, "r")
  sims = []
  for line in fin:
    line = line.strip()
    pieces = line.split(",")
    sims.append([float(f) for f in pieces])
  fin.close()
  return sims

def merge_sims(sims1, sims2, weight1, weight2):
  assert len(sims1) == len(sims2)
  newsims = []
  for i in range(len(sims1)):
    sims = []
    for j in range(len(sims1[i])):
      nsim = sims1[i][j]*weight1 + sims2[i][j]*weight2
      sims.append(nsim)
    newsims.append(sims)
  return newsims

def main():
  if len(sys.argv) != 6:
    print("using: %s simfile1 simfile2 outfile weight1 weight2" % sys.argv[0])
    exit()
  sim1 = get_sim(sys.argv[1])
  sim2 = get_sim(sys.argv[2])
  outfile = sys.argv[3]
  weight1 = float(sys.argv[4])
  weight2 = float(sys.argv[5])
  newsims = merge_sims(sim1, sim2, weight1, weight2)
  fout = open(outfile, "w")
  for sims in newsims:
    fout.write(",".join([str(sim) for sim in sims]))
    fout.write("\n")
  fout.close()

if __name__ == "__main__":
  main()

