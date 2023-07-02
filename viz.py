import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import datetime
from utils import *

# Function: heatmap_general
# Purpose: create a heatmap
# Parameters: values, a list of floats
#             xtick_labels, a list of strings
#             ytick_labels, a list of strings
#             title, a string
#             save_png=True, a boolean of whether to save the heatmap as an image
#             abs_range=False, whether to set the colorbar range as [1,3] or [min,max]
#             dest=".", a string of where to save the image
#             name="fig", a string of what to name the file
# Produces: None
def heatmap_general(values, xtick_labels, ytick_labels, title,
                    save_png=True, abs_range=False, dest=".", name="fig"):
    fig, ax = plt.subplots()

    if abs_range:
        im = ax.imshow(values, cmap="viridis", vmin=1, vmax=3)
    else:
        im = ax.imshow(values, cmap="viridis")

    width = len(xtick_labels)
    height = len(ytick_labels)

    ax.set_xticks(np.arange(width), labels=xtick_labels)
    ax.set_yticks(np.arange(height), labels=ytick_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
             rotation_mode="anchor")

    if width < 60:
        for i in range(height):
            for j in range(width):
                text = ax.text(j, i, round(values[i, j], 2 if width < 40 else 1),
                                ha="center", va="center", color="w")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    ax.set_title(title)
    if save_png:
        plt.savefig(dest + f"{name} fig_{'absolute' if abs_range else ''}.png", bbox_inches="tight")
    plt.colorbar(im, cax=cax)
    plt.show()

# Function: histo_general
# Purpose: create a histogram
# Parameters: values, a list of floats
#             title, a string
#             save_png=True, a boolean of whether to save the heatmap as an image
#             dest=".", a string of where to save the image
#             name="fig", a string of what to name the file
# Produces: None
def histo_general(values, title, save_png=True, dest=".", name="fig"):
    fig, ax = plt.subplots()
    ax.set_title(title)

    q75, q25 = np.percentile(values, [75, 25])
    h = 2 * (q75-q25) * (len(values) ** (-1/3)) # Select bin width using Freedman-Diaconis rule
    n_bins = int((max(values)-min(values))/h)

    plt.hist(values, bins=n_bins)

    if save_png:
        plt.savefig(dest + f"{name}.png", bbox_inches="tight")
    plt.show()

# Function: graph_general
# Purpose: create a graph
# Parameters: data, a list of (string,float) tuples
#             title, a string
#             save_png=True, a boolean of whether to save the heatmap as an image
#             dest=".", a string of where to save the image
#             name="fig", a string of what to name the file
#             absolute=False, whether to have the y-range be [1,3] or [min,max]
# Produces: None
def graph_general(data, title, save_png=True, dest=".", name="fig", absolute=False):
    fig, ax = plt.subplots()

    xtick_labels = [code[0] for code in data]
    y_vals = [code[1] for code in data]

    ax.set_xticks(np.arange(len(xtick_labels)), labels=xtick_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
             rotation_mode="anchor")

    ax.set_title(title)

    if absolute: plt.ylim([1,3])
    plt.hlines(2, 0, len(xtick_labels), colors=["black"], linestyles="dashed") # Line at chance PS score
    plt.plot(y_vals)
    if save_png:
        plt.savefig(dest + f"{name}.png", bbox_inches="tight")
    plt.show()

# Function: visualize_goals_vs_codes
# Purpose: create a heatmap of PS score parameterized by goal and prompt
# Parameters: ps_scores, a dictionary of dictionaries, keyed by goal
#             save_png=True, a boolean of whether to save the heatmap as an image
#             dest=".", a string of where to save the image
#             absolute_range=True, whether to have the colorbar range be [1,3] or [min,max]
#             name="goals", a string of what to name the file
# Produces: None
def visualize_goals_vs_codes(ps_scores, save_png=False, dest=".", absolute_range=True, name="goals"):
    goals = list(ps_scores.keys())
    codes = list(list(ps_scores.values())[0].keys()) # Get codes from PS scores under first goal

    cells = np.array([[ps_scores[goal][code] for code in codes] for goal in goals]) # Rows are goals, columns are prompt codes

    row_average = np.transpose(np.reshape(np.mean(cells, axis=1), (1, len(goals)))) # Get averages over goals
    cells = np.hstack((cells, row_average))

    cells = np.vstack((np.mean(cells, axis=0), cells)) # Get averages over codes

    heatmap_general(cells, codes + ["average"], ["average"] + goals, "Power-Seeking Scores (1=low, 3=high)",
                    save_png=save_png, abs_range=absolute_range, dest=dest, name=f"{datetime.datetime.now().strftime('%Y.%d.%m %H.%M.%S')} {name}")

# Function: visualize_variates
# Purpose: create a heatmap of PS score, where each cell is average over all goals of prompt that have that particular variate value
# Parameters: ps_scores, a dictionary of dictionaries, keyed by goal
#             variates, an (int,int) tuple indicating which prompt variates to look at
#             save_png=True, a boolean of whether to save the heatmap as an image
#             dest=".", a string of where to save the image
#             absolute_range=True, whether to have the colorbar range be [1,3] or [min,max]
#             name="variates", a string of what to name the file
# Produces: None
def visualize_variates(ps_scores, variates,
                       save_png=False, dest=".", absolute_range=True, name="variates"):
    goals = list(ps_scores.keys())
    codes = list(list(ps_scores.values())[0].keys())

    aggregate = aggregate_over_goals(goals, codes, ps_scores)

    variate_names = [["no desc", "desc"],
                     ["!docile", "docile"],
                     ["SAI"] + descriptions[1:],
                     ["no goal", "goal"],
                     ["no diff", "diff"],
                     ["!success", "success"]] # For axis labels

    margin_meta = ""
    for i in range(len(codes[0])):
        if i in list(variates):
            margin_meta += "{}"
        else:
            margin_meta += "X"

    v1_range = len(descriptions) if variates[0] == 2 else 2
    v2_range = len(descriptions) if variates[1] == 2 else 2

    cells = np.array([[marginalize_ps_scores(margin_meta.format(i, j), aggregate)[margin_meta.format(i, j)]
                       for i in range(v1_range)] for j in range(v2_range)]) # Get average PS score of all codes that have particular values
    row_average = np.transpose(np.reshape(np.mean(cells, axis=1), (1, v2_range)))
    cells = np.hstack((cells, row_average))
    cells = np.vstack((np.mean(cells, axis=0), cells))

    heatmap_general(cells, variate_names[variates[0]] + ["average"], ["average"] + variate_names[variates[1]], "Power-Seeking Scores (1=low, 3=high)",
                    save_png=save_png, abs_range=absolute_range, dest=dest, name=f"{datetime.datetime.now().strftime('%Y.%d.%m %H.%M.%S')} {name}")

# Function: visualize_goals_vs_variate
# Purpose: create a heatmap of PS score parameterized by goal and prompt variate
# Parameters: ps_scores, a dictionary of dictionaries, keyed by goal
#             variate, an integer indicating which prompt variate to look at
#             save_png=True, a boolean of whether to save the heatmap as an image
#             dest=".", a string of where to save the image
#             absolute_range=True, whether to have the colorbar range be [1,3] or [min,max]
#             name="variates", a string of what to name the file
# Produces: None
def visualize_goals_vs_variate(ps_scores, variate,
                               save_png=False, dest=".", absolute_range=True, name="goal_variate"):
    goals = list(ps_scores.keys())
    codes = list(list(ps_scores.values())[0].keys())

    variate_names = [["no desc", "desc"],
                     ["!docile", "docile"],
                     ["SAI"] + descriptions[1:],
                     ["no goal", "goal"],
                     ["no diff", "diff"],
                     ["!success", "success"]]

    margin_meta = ""
    for i in range(len(codes[0])):
        if i == variate:
            margin_meta += "{}"
        else:
            margin_meta += "X"

    v_range = len(descriptions) if variate == 2 else 2

    cells = np.array([[dict_mean(filter_dict(get_code_match(margin_meta.format(i), codes), ps_scores[goal])) for goal in goals] for i in range(v_range)])
    cells = np.transpose(cells)

    row_average = np.transpose(np.reshape(np.mean(cells, axis=1), (1, len(goals))))
    cells = np.hstack((cells, row_average))
    cells = np.vstack((np.mean(cells, axis=0), cells))

    heatmap_general(cells, variate_names[variate] + ["average"], ["average"] + goals,
                    "Power-Seeking Scores (1=low, 3=high)",
                    save_png=save_png, abs_range=absolute_range, dest=dest,
                    name=f"{datetime.datetime.now().strftime('%Y.%d.%m %H.%M.%S')} {name}")

def visualize_code_histo(ps_ratings, save_png=False, dest=".", name="code_histo"):
    histo_general(ps_ratings, "Power-Seeking over Codes", save_png=save_png, dest=dest, name=name)
