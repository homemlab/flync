import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set_theme(style="white")
cutoff=int(sys.argv[1])
workdir=(sys.argv[2])

mapstats = pd.read_csv(workdir + '/results/map-stats.tsv', delimiter='\t')
runinfo = pd.read_csv(workdir + '/results/runinfo.csv')

mapstats.sort_values(by=['sra_access'])
runinfo.sort_values(by=['sra_access'])

bio_proj = np.where(runinfo['sra_access'] == mapstats['sra_access'], runinfo['bio_proj'], np.nan)
mapstats.insert(0, 'bio_proj', bio_proj)

### Define cutoff for map quality. Below <int> % of total read algined to reference genome will be considered Bad.
mapstats['map_qual'] = np.where((mapstats['percent_aligned']<cutoff), 'Bad', 'Good')

fig, ax = plt.subplots()
ax.set_xlim(0,100)

map_qual_plot = sns.scatterplot(ax=ax, x="percent_aligned", y="sra_access", style="map_qual", size="num_reads", hue="bio_proj", legend='brief', alpha=.5, data=mapstats)
plt.savefig(workdir + '/results/map_qual_plot.png')

sra_include = mapstats[mapstats.map_qual == 'Good']
sra_include = sra_include['sra_access']
textfile = open(workdir + "/results/filtered_list.txt", "w")

for element in sra_include:
    textfile.write(element + "\n")
textfile.close()