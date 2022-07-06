# EdgePhase

- EdgeConv [package of EdgePhase]
  - DataGeneratorMulti.py [data generator]
  - Utils.py [function of picking P/S phases based on the  probability]
  - trainerBaselineA.py, trainerEdgePhase.py

- Example [the case study of Greece Earthqquake]
  -  catalog_visualization 
      - doc.kml [fault data of Greece]
      -  NKUA_cat.csv [catalog of NKUA]
      -  EdgePhase_cat.csv [catalog of EdgePhase]
      -  catalog_visualization.ipynb [notebook for visualization]
  -  EqLocation [softwares/control files for earthquake location]
      -  REAL
      -  VELEST
      -  HypoDD
  - download.py, preprocess.py, run_EdgePhase.py, pick_phase.py [scripts to use EdgePhase for case study]
  - magnitude_estimation.py [magnitude estimation code]
  - station.csv, edge_index.pt [station data]
 
- SCSN2021Dataset [code to build training/validation dataset]
- fine-tune_plot [code to visualize the model performance of validation set]
- models [contains the weights of trained models]

- EdgePhase_tutorial.ipynb [a tutorial on google colab]
- finetune_baselineA.py, finetune_edgephase.py, test_edgephase.py [scripts to train/test models]
