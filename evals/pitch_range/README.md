## Reproduce the pitch range evaluations

We use [RMVPE](https://github.com/Dream-High/RMVPE) to extract the raw pitch values. Please see `extract_pitch_values_from_audio/main.py` for the extraction script.
Before running the extraction script, please download the [pretrained RMVPE checkpoint](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip).
The extraction script will copy all folder structure from input directory, but replace all audio file with corresponding pitch values, one line per value.
We provide already extracted pitch values in `raw_pitch_extracted` and the system-level concatenated values in `raw_pitch_extracted_combined`. See `raw_pitch_extracted/concat_txt.sh` for script used to concatenate text files. To understand the system-level behavior better, please see `raw_pitch_extracted_combined/analyze_f0.py`.

See `plot_violin_plot.py` for the script used to plot the violin plot for vocal range distribution. Before running, please replace `root_directory` to the correct path.