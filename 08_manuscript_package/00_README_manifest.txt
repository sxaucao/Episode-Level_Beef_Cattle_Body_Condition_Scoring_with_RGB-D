Animals manuscript package
==========================

Generated at:
2026-09-17 11:00:48

Base path:
.

Output package:
08_manuscript_package

Final result rule
-----------------
The main manuscript result is based on:
  04_full_ablation_results/qg_rgb_depth_lowhigh_with_angle

Final model:
  Quality-gated RGB-depth LowHigh CO-MIL with QC dorsal angle


Folder guide
------------
01_figures_main/
  Main paper figures in manuscript order.
  Figure 1, Figure 2, and Figure 4 may still need manual design if missing.

02_tables_main/
  Main paper tables in manuscript order.

03_supplementary/
  Detailed model reports, predictions, supplementary angle tests, and raw ablation outputs.

04_missing_or_manual_items/
  Checklist for figures that usually require manual drawing/assembly.

Recommended main figures
------------------------
Figure01: Overall workflow diagram                     [manual if missing]
Figure02: Good/bad frame quality-control examples      [manual if missing]
Figure03: Dataset distribution and good-frame ratio
Figure04: Model architecture diagram                   [manual if missing]
Figure05: Final model confusion matrices
Figure06: QC dorsal angle statistics
Figure07: Model performance comparison
Figure08: Extreme error comparison

Recommended main tables
-----------------------
Table01: Dataset summary
Table02: Animal-wise fold distribution and QC coverage
Table03: Final main model results
Table04: Full ablation study
Table05: Extreme error comparison
Table06: QC dorsal angle statistics
Table07: Contextual comparison with reference LACO ViT

Check:
  00_manifest_copied_files.csv
  00_missing_files.csv