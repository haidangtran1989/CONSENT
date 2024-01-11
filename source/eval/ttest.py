import sys

from scipy.stats import ttest_rel


def extract_output(file_name, target_group):
    with open(file_name, "rt", encoding="utf-8") as file_output:
        content = file_output.read()
    lines = content.strip().split("\n")
    ret = list()
    for line in lines:
        parts = line.split(" ")
        pop = int(parts[2])
        if pop < 0:
            group = "ookb"
        elif pop < 90:
            group = "rare"
        else:
            group = "uncommon"
        metric = float(parts[3])
        if target_group == "all" or group == target_group:
            ret.append(metric)
    return ret


x = extract_output(sys.argv[1], sys.argv[3])
y = extract_output(sys.argv[2], sys.argv[3])

pvalue = ttest_rel(x, y).pvalue
print("average x = ", sum(x) / len(x))
print("average y = ", sum(y) / len(y))
print("pvalue = " + str(pvalue), "len = ", len(x))
