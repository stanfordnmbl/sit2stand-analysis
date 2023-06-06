# Sit2stand.ai -- scripts replicating data analaysis in the paper

## Quickstart

### Extraction of movement metrics from trajectories (reproducing our paper)
1. Install jupyter notebook and requirements with
```
pip install -r requirements.txt
```
2. Download processed videos (trajectories of body landmarks) from [Google Drive](https://drive.google.com/file/d/1fJIOuo5tfJ9SNLkL0wXXM58ekVURvorS/view?usp=sharing) and unzip them to the `videos/np/` directory
3. Run `GetMetrics.ipynb` to derive all metrics used for statistical analysis -- they will be saved in `results.csv`

### Data analysis
1. Start an rstudio project in stats directory
2. Run `sit2stand_clean-data_v15.Rmd` notebook with `results.csv` derived previously or an already provided `dataClean.csv` file.

### Optional: Processing videos
If you don't want to use our preprocessed video trajectories, you can process videos on your own. Note that results may be slightly different from ours since we only share deidentified videos, while we ran open pose on raw videos.

1. Download videos from our [Google Drive](https://drive.google.com/drive/folders/1C6777-AFWU2LUPx9ydqtuJjqrMi2oDwS?usp=sharing).
2. Run openpose on videos, for example as we did [here](https://github.com/stanfordnmbl/sit2stand/blob/main/processing-docker/demo.py#L25)
3. Process videos to get x,y trajectories of keypoints and save them as numpy arrays as we did [here](https://github.com/stanfordnmbl/sit2stand-analysis/blob/main/PrepareData.ipynb)
