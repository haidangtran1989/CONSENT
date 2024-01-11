import sys

file_name = sys.argv[1]
metric_by_turn = dict()
with open(file_name, "rt", encoding="utf-8") as file:
    content = file.read().strip()
    lines = content.split("\n")[1:-3]
    for line in lines:
        parts = line.split("\t")
        metric = float(parts[1])
        q_id = parts[0]
        turn_number = int(q_id.split("_")[-1]) + 1
        metrics = metric_by_turn.get(turn_number, list())
        metrics.append(metric)
        metric_by_turn[turn_number] = metrics

for turn_number, metrics in metric_by_turn.items():
    metric_avg = sum(metrics) / len(metrics)
    print(str(turn_number) + "\t" + str(metric_avg) + "\t" + str(len(metrics)))
