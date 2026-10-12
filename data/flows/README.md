# NFL flow weights (2D2H2L text format), our own

- `<ds>_s512_t2000.txt` (fb, osm, planet) and `<ds>_s64_t2000.txt` (the other seven): the `SCALELI_FLOW` files
  `scripts/gre_run.sh` passes to the scaleli_n/_nc/_j/_jg0 cells (`period()`: s512 for fb, osm and planet, s64
  otherwise). Trained on the Mac by `main@ca8d80e:experimental/scaleli/tools/train_flow.py
  data/samples/<ds>_2M_uniform_s42 --dtype uint64 --sample 16384
  --steps 2000 --shifts 64|512 --lr 0.05` (main@ca8d80e:experimental/scaleli/results/aidb_flowv2/retrain.sh; the
  trainer, the logs and the unused shift are on main).
- `dense_sparse_2D2H2L.txt`: the test flow of ctest `scaleli_hardness_tests` (20,000 synthetic dense_sparse keys,
  200 steps; main@ca8d80e:experimental/scaleli/results/flows/training_report.json).
- All are our own weights from a dependency-free stand-in trainer (1-D change-of-variables likelihood, Adam; not a
  BNAF), in the format NFL exports; not NFL's author models.
