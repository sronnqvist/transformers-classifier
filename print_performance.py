import sys
import csv
import collections
import numpy as np


data = collections.defaultdict(lambda: [])
for filename in sys.argv[1:]:
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            setting = tuple(((k,row[k]) for k in row.keys() if k.startswith('p') or k == '_Epoch'))
            results = {k: row[k] for k in row.keys() if not (k.startswith('p') or k == '_Epoch')}
            data[setting].append(results)

mean = (lambda x: sum(x)/len(x))
try:
    ranking = [(np.mean([float(x['_f1']) for x in data[setting]]),
                   np.mean([float(x['_val_auc']) for x in data[setting]]),
                   setting) for setting in data]
except KeyError:
    ranking = [(np.mean([float(x['_f1']) for x in data[setting]]),
                   setting) for setting in data]

print("Best settings by validation F-score:")

"""for i, (f1, auc, setting) in enumerate(sorted(ranking, key=lambda x: -x[0])[:5]):
    print("%d." % (i+1), end="")
    print(" F1:", f1)
    print("   Settings:", ', '.join(["%s:%s" % x for x in setting]))
    print("   N experiments:", len(data[setting]))
    #for k in data[setting][0].keys():
    #    print("%s: %s" % (k, mean([float(x[k]) for x in data[setting]])))
    print()


print("Best settings by validation AUC:")
for i, (f1, auc, setting) in enumerate(sorted(ranking, key=lambda x: -x[1])[:5]):
    print("%d." % (i+1), end="")
    print(" AUC:", auc)
    print("   Settings:", ', '.join(["%s:%s" % x for x in setting]))
    print("   N experiments:", len(data[setting]))
    print()
"""
print("F1\t(sd)\tAUC\t(sd)\t%s\tNexp" % '\t'.join([x for x,_ in ranking[0][-1]]))
for i, x in enumerate(sorted(ranking, key=lambda x: -x[0])[:20]):
    if len(x) == 3:
        f1, auc, setting = x
        print("%.4f (%.4f)\t%.4f (%.4f)\t%s\t%s" % (f1, np.std([float(x['_f1']) for x in data[setting]]), auc, np.std([float(x['_val_auc']) for x in data[setting]]), '\t'.join([x for _,x in setting]), len(data[setting])))
    else:
        f1, setting = x
        print("%.4f (%.4f)\t--- (---)\t%s\t%s" % (f1, np.std([float(x['_f1']) for x in data[setting]]), '\t'.join([x for _,x in setting]), len(data[setting])))

print()
if len(ranking[0]) == 3:
    for i, x in enumerate(sorted(ranking, key=lambda x: -x[1])[:10]):
        f1, auc, setting = x
        print("%.4f (%.4f)\t%.4f (%.4f)\t%s\t%s" % (f1, np.std([float(x['_f1']) for x in data[setting]]), auc, np.std([float(x['_val_auc']) for x in data[setting]]), '\t'.join([x for _,x in setting]), len(data[setting])))
    #print("%.4f\t%.4f\t%.4f\t%.4f\t%s\t%s" % (f1, auc, '\t'.join([x for _,x in setting]), len(data[setting])))
