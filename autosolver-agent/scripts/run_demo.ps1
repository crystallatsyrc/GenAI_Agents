param(
  [string]$Case = "examples\tiny_case.tsv",
  [double]$TimeLimit = 1.0
)

python autosolver\local_judge.py --solver solver.py --case $Case --time-limit $TimeLimit
