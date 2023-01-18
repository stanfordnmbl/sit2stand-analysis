# Sit2stand.ai -- scripts replicating data analaysis in the paper

## Quickstart

Download videos from our [Google Drive](https://drive.google.com/drive/folders/1C6777-AFWU2LUPx9ydqtuJjqrMi2oDwS?usp=sharing).

Extraction of movement metrics
1. Install jupyter notebook and requirements with
```
pip install -r requirements.txt
```
2. Run `PrepareData.ipynb` to download keypoints of all subjects
3. Run `GetMetrics.ipynb` to derive all metrics used for statistical analysis -- they will be saved in `results.csv`

Data analysis
1. Start an rstudio project in stats directory
2. Run `sit2stand_clean-data_v15.Rmd` notebook with `results.csv` derived previously or an already provided `dataClean.csv` file.
