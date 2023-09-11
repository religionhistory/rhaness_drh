import sys
import textwrap

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import nan_euclidean_distances

### Globals
FONT_SIZE = 18
DIRECTORY = "figures"
CMAP = colors.LinearSegmentedColormap.from_list("", ["white", "black"])


### Utils
def leaf_label_func(id):
    return entry_names[id]


def jaccard_similarity(df):
    # Create an empty DataFrame for the similarity matrix
    similarity_matrix = pd.DataFrame(index=df.index, columns=df.index)

    # Calculate the Jaccard similarity for each pair of rows
    for i in df.index:
        for j in df.index:
            if i != j:
                pair_data_i = df.loc[i].dropna()
                pair_data_j = df.loc[j].dropna()

                intersection = sum(pair_data_i.index.isin(pair_data_j.index))
                union = len(np.union1d(pair_data_i.index, pair_data_j.index))

                # If union is 0 (which means both rows have only NaNs),
                # handle it specifically to avoid ZeroDivisionError
                if union == 0:
                    similarity_matrix.loc[i, j] = np.nan
                else:
                    similarity_matrix.loc[i, j] = intersection / union
            else:
                similarity_matrix.loc[i, j] = 1
    return similarity_matrix


### Load Data
df = pd.read_csv("raw_data.csv", index_col=0)
grouped = df.groupby("region")
all_entries_by_region = {}
for region_name, region_entries in grouped:
    all_entries_by_region[region_name] = region_entries.copy()


## Produce Figures

## Figure 4 - Kendall Tau correlation of Group Entries
fig_4_name = "Kendall Tau correlation of Group Entries"
print("Figure 4: ", fig_4_name)

# Subset data
fig_4_data = df[df["poll"] == "group"]

# Check for duplicate entries and remove
duplicates = fig_4_data.index.duplicated(keep="first")
fig_4_data = fig_4_data[~duplicates]

# Produce Kendall Tau correlation
corr = fig_4_data[
    [col for col in fig_4_data.columns if col not in ["name", "region", "poll"]]
].T.corr(method="kendall")

# Make sure NA's are really NA's
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
corr[mask] = np.NaN

# Create figure
f = plt.figure(figsize=(50, 50))
f.set_facecolor("white")
plt.matshow(corr, cmap=CMAP, fignum=f.number)
plt.autoscale()
entry_names = [
    "\n".join(textwrap.wrap(name, 25)) for name in fig_4_data["name"].tolist()
]
plt.gca().xaxis.tick_bottom()
label_font_size = FONT_SIZE - 2
locs, labels = plt.xticks(
    range(len(entry_names)),
    entry_names,
    fontsize=label_font_size,
    rotation=45,
    horizontalalignment="right",
)
plt.yticks(range(len(entry_names)), entry_names, fontsize=label_font_size)
plt.title(fig_4_name, fontsize=FONT_SIZE * 3, pad=100)
plt.legend
plt.savefig(f"{DIRECTORY}/figure_4.png", bbox_inches="tight")
plt.close()


## Figure 5 - Kendall Tau correlation of Place Entries
fig_5_name = "Kendall Tau correlation of Place Entries"
print("Figure 5: ", fig_5_name)

# Subset data
fig_5_data = df[df["poll"] == "place"]

# Check for duplicate entries and remove
duplicates = fig_5_data.index.duplicated(keep="first")
fig_5_data = fig_5_data[~duplicates]

# Produce Kendall Tau correlation
corr = fig_5_data[
    [col for col in fig_5_data.columns if col not in ["name", "region", "poll"]]
].T.corr(method="kendall")

# Make sure NA's are really NA's
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
corr[mask] = np.NaN

# Create figure
f = plt.figure(figsize=(50, 50))
f.set_facecolor("white")
plt.matshow(corr, cmap=CMAP, fignum=f.number)
plt.autoscale()
entry_names = [
    "\n".join(textwrap.wrap(name, 25)) for name in fig_5_data["name"].tolist()
]
plt.gca().xaxis.tick_bottom()
label_font_size = FONT_SIZE
locs, labels = plt.xticks(
    range(len(entry_names)),
    entry_names,
    fontsize=label_font_size,
    rotation=45,
    horizontalalignment="right",
)
plt.yticks(range(len(entry_names)), entry_names, fontsize=label_font_size)
plt.title(fig_5_name, fontsize=FONT_SIZE * 3, pad=100)
plt.legend
plt.savefig(f"{DIRECTORY}/figure_5.png", bbox_inches="tight")
plt.close()


## Figure 6 - Kendall Tau correlation of Text entries
fig_6_name = "Kendall Tau correlation of Text entries"
print("Figure 6: ", fig_6_name)

# Subset data
fig_6_data = df[df["poll"] == "text"]

# Check for duplicate entries and remove
duplicates = fig_6_data.index.duplicated(keep="first")
fig_6_data = fig_6_data[~duplicates]

# Produce Kendall Tau correlation
corr = fig_6_data[
    [col for col in fig_6_data.columns if col not in ["name", "region", "poll"]]
].T.corr(method="kendall")

# Make sure NA's are really NA's
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
corr[mask] = np.NaN

# Create figure
f = plt.figure(figsize=(50, 50))
f.set_facecolor("white")
plt.matshow(corr, cmap=CMAP, fignum=f.number)
plt.autoscale()
entry_names = [
    "\n".join(textwrap.wrap(name, 25)) for name in fig_6_data["name"].tolist()
]
plt.gca().xaxis.tick_bottom()
label_font_size = FONT_SIZE
locs, labels = plt.xticks(
    range(len(entry_names)),
    entry_names,
    fontsize=label_font_size,
    rotation=45,
    horizontalalignment="right",
)
plt.yticks(range(len(entry_names)), entry_names, fontsize=label_font_size)
plt.title(fig_6_name, fontsize=FONT_SIZE * 3, pad=100)
plt.legend
plt.savefig(f"{DIRECTORY}/figure_6.png", bbox_inches="tight")
plt.close()


## Figure 7 - Jaccard Index on Mesopotamian Group entries
fig_7_name = "Jaccard Index on Mesopotamian Group entries"
print("Figure 7: ", fig_7_name)

# Subset data
fig_7_data = df[df["poll"] == "group"]
fig_7_data = fig_7_data[fig_7_data["region"] == "mesopotamia"]

# Check for duplicate entries and remove
duplicates = fig_7_data.index.duplicated(keep="first")
fig_7_data = fig_7_data[~duplicates]

# Produce Kendall Tau correlation
corr = jaccard_similarity(
    fig_7_data[
        [col for col in fig_7_data.columns if col not in ["name", "region", "poll"]]
    ]
).astype(float)

# Make sure NA's are really NA's
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
corr[mask] = np.NaN

# Create figure
f = plt.figure(figsize=(50, 50))
f.set_facecolor("white")
plt.matshow(corr, cmap=CMAP, fignum=f.number)
plt.autoscale()
entry_names = [
    "\n".join(textwrap.wrap(name, 25)) for name in fig_7_data["name"].tolist()
]
plt.gca().xaxis.tick_bottom()
label_font_size = FONT_SIZE * 2
locs, labels = plt.xticks(
    range(len(entry_names)),
    entry_names,
    fontsize=label_font_size,
    rotation=45,
    horizontalalignment="right",
)
plt.yticks(range(len(entry_names)), entry_names, fontsize=label_font_size)
plt.title(fig_7_name, fontsize=FONT_SIZE * 3, pad=100)
plt.legend
plt.savefig(f"{DIRECTORY}/figure_7.png", bbox_inches="tight")
plt.close()


## Figure 8 - Euclidean Distance on Levantine Place entries
fig_8_name = "Euclidean Distance on Levantine Place entries"
print("Figure 8: ", fig_8_name)

# Subset data
fig_8_data = df[df["poll"] == "place"]
fig_8_data = fig_8_data[fig_8_data["region"] == "levant"]

# Check for duplicate entries and remove
duplicates = fig_8_data.index.duplicated(keep="first")
fig_8_data = fig_8_data[~duplicates]

# Produce Kendall Tau correlation
corr = pd.DataFrame(
    1
    - nan_euclidean_distances(
        fig_8_data[
            [col for col in fig_8_data.columns if col not in ["name", "region", "poll"]]
        ]
    )
)

# Make sure NA's are really NA's
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
corr[mask] = np.NaN

# Create figure
f = plt.figure(figsize=(50, 50))
f.set_facecolor("white")
plt.matshow(corr, cmap=CMAP, fignum=f.number)
plt.autoscale()
entry_names = [
    "\n".join(textwrap.wrap(name, 25)) for name in fig_8_data["name"].tolist()
]
plt.gca().xaxis.tick_bottom()
label_font_size = FONT_SIZE
locs, labels = plt.xticks(
    range(len(entry_names)),
    entry_names,
    fontsize=label_font_size,
    rotation=45,
    horizontalalignment="right",
)
plt.yticks(range(len(entry_names)), entry_names, fontsize=label_font_size)
plt.title(fig_8_name, fontsize=FONT_SIZE * 3, pad=100)
plt.legend
plt.savefig(f"{DIRECTORY}/figure_8.png", bbox_inches="tight")
plt.close()


### Dendrogram
# Figure 9 - Dendrogram of Levantine Place entries
fig_9_name = "Dendrogram of Levantine Place entries"
print("Figure 9: ", fig_9_name)

# Subset data
fig_9_data = df[df["poll"] == "place"]
fig_9_data = fig_9_data[fig_9_data["region"] == "levant"]

# Check for duplicate entries and remove
duplicates = fig_9_data.index.duplicated(keep="first")
fig_9_data = fig_9_data[~duplicates]

corr = fig_9_data[
    [col for col in fig_9_data.columns if col not in ["name", "region", "poll"]]
].T.corr(method="kendall")

entry_names = [
    "\n".join(textwrap.wrap(name, 25)) for name in fig_9_data["name"].tolist()
]
pairwise_distance = pdist(corr)
clustering = linkage(pairwise_distance, method="average")

# Create figure
f = plt.figure(figsize=(50, 50))
f.set_facecolor("white")
label_font_size = FONT_SIZE * 2
with plt.rc_context({"lines.linewidth": 5}):
    dendrogram(
        clustering,
        orientation="right",
        leaf_label_func=leaf_label_func,
        leaf_font_size=label_font_size,
    )

plt.title(fig_9_name, fontsize=FONT_SIZE * 3, pad=100)
plt.legend
# plt.tight_layout()
plt.savefig(f"{DIRECTORY}/figure_9.png", bbox_inches="tight")
plt.close()
