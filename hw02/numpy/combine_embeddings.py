import numpy as np

def load_vectors(fname):
    words = []
    vecs = []
    with open(fname, "r") as f:
        header = f.readline().strip().split()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            words.append(parts[0])
            vecs.append([float(x) for x in parts[1:]])
    return words, np.array(vecs, dtype=np.float32)

def save_vectors(words, vecs, out):
    with open(out, "w") as f:
        f.write(f"{len(words)} {vecs.shape[1]}\n")
        for w, v in zip(words, vecs):
            f.write(w + " " + " ".join(map(str, v)) + "\n")

def combine(tfile, cfile, out):
    words1, et = load_vectors(tfile)
    words2, ec = load_vectors(cfile)

    assert words1 == words2, "Word orders must match"

    comb = et + ec
    save_vectors(words1, comb, out)

if __name__ == "__main__":
    combine("vec.ctx5.txt.1", "vec.ctx5.txt.2", "vec.ctx5.combined.txt")
    combine("vec.ctx15.txt.1", "vec.ctx15.txt.2", "vec.ctx15.combined.txt")
    print("Combined embeddings created.")