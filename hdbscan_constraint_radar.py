# -*- coding: utf-8 -*-
"""
@author: Claudia Malzer 2021
"""

import numpy as np
import os
import pandas as pd
import hdbscan_constraint
import hdbscan
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
from sklearn import metrics
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
sns.set_color_codes()
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

plt.rc('font', size=16)
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=16)

fileLocation = os.path.realpath(__file__)
workspace = os.path.dirname(fileLocation)
basePath = os.path.join(workspace, "nuscenes_data_labeled")

HDBSCAN_CONSTRAINT = "HDBSCAN(constraint)"
HDBSCAN_CE = "HDBSCAN(c+eps)"
HDBSCAN_EC = "HDBSCAN(eps+c)"
HDBSCAN_B3F = "HDBSCAN(b3f)"
HDBSCAN_EOM = "HDBSCAN(eom)"
DBSCAN_STAR = "DBSCAN*"

CONFIG = {
    "plot_results" : False,
    "plot_true_labels": False,
    "sample_fraction": 0.1, #set to 1 for best theoretically achievable result
    "evaluate_alternatives": False,
    "plot_tree": False #requires pygraphviz (https://pygraphviz.github.io/documentation/stable/install.html)
  }

def cluster_radar_data(coords, prelabels, velocities, directions, xy, epsilon, cluster_mode, title):

    minPts = 2
    print("MinPts: {}, Epsilon: {}, Number of points: {}".format(minPts, epsilon, len(coords)))
    allow_single_cluster = True

    if cluster_mode == HDBSCAN_B3F:
        print("Clustering using HDBSCAN(b3f) with prelabels...")
        clusterer = hdbscan_constraint.HDBSCAN(min_cluster_size=minPts,cluster_selection_epsilon=epsilon, cluster_selection_method='b3f', allow_single_cluster=allow_single_cluster, prelabels=prelabels, velocities=velocities, xy=xy, directions=directions)

    elif cluster_mode == HDBSCAN_EOM:
        print("Clustering using default HDBSCAN(eom) with epsilon = {}".format(epsilon))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=minPts, cluster_selection_epsilon=epsilon, cluster_selection_method='eom', allow_single_cluster=allow_single_cluster)

    elif cluster_mode == DBSCAN_STAR:
        print("Clustering using default DBSCAN* with epsilon = {}".format(epsilon))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=minPts).fit(coords)

    elif cluster_mode in (HDBSCAN_CONSTRAINT, HDBSCAN_CE, HDBSCAN_EC):
        print("Clustering using HDBSCAN(constraint)...")
        selection_method = 'constraint+e'
        if cluster_mode == HDBSCAN_EC:
            selection_method='e+constraint'
        clusterer = hdbscan_constraint.HDBSCAN(min_cluster_size=minPts, cluster_selection_epsilon=epsilon, cluster_selection_method=selection_method, allow_single_cluster=allow_single_cluster, prelabels=prelabels, velocities=velocities, xy=xy, directions=directions)

    alternatives = None
    if cluster_mode in (HDBSCAN_B3F, HDBSCAN_CONSTRAINT, HDBSCAN_CE, HDBSCAN_EC):
        cluster_labels, alternatives = clusterer.fit_predict(coords)

        if CONFIG['plot_tree'] and cluster_mode in (HDBSCAN_CONSTRAINT, HDBSCAN_B3F, HDBSCAN_CE, HDBSCAN_EC):
            df = clusterer.condensed_tree_.to_pandas()
            df = df[df['child_size'] > 1]
            visualize_tree(df, title)

    elif cluster_mode == DBSCAN_STAR:
        cluster_labels = clusterer.single_linkage_tree_.get_clusters(epsilon, min_cluster_size=minPts)

    else:
        cluster_labels = clusterer.fit_predict(coords)

    noise = list(cluster_labels).count(-1)
    print("{} clusters detected using {} with epsilon = {}, {} points declared noise.".format(cluster_labels.max()+1, cluster_mode, epsilon, noise))
    return (cluster_labels, alternatives)

def noise_to_singleton(labels, maxlabel):
    new_labels = list()
    increment = -1 if maxlabel > 0 else 1
    for x in labels:
        if x == -1:
            new_labels.append(maxlabel)
            maxlabel=maxlabel+(-1*increment)
        else:
            new_labels.append(x)
    return new_labels

def evaluate(cluster_labels, radar_df):

    radar_df['cluster_labels'] = cluster_labels
    max_true_label = radar_df['label'].max()+1
    max_noise_label = -1
    true_labels = noise_to_singleton(radar_df['label'].values, max_true_label)
    new_cluster_labels = noise_to_singleton(cluster_labels, max_noise_label)
    ari = metrics.adjusted_rand_score(true_labels, new_cluster_labels)
    return (ari, radar_df)

def get_true_labels_percentage(radar_df, true_labels, percentage, seed):

     if percentage == 1:
        return true_labels

     radar_df['true_labels'] = true_labels
     radar_df['samples'] = -1
     labels_without_noise = radar_df[radar_df['true_labels'] != -1]['true_labels']
     samples = labels_without_noise.sample(frac=percentage, replace=False, random_state=seed)
     if len(samples) < 2:
         samples = labels_without_noise.sample(n=2, replace=False, random_state=seed)
     radar_df['samples'].update(samples)

     return radar_df['samples'].values


def visualize_tree(df, title):

    G = nx.DiGraph()
    plt.figure(figsize=(12,10), dpi=144)
    ax = plt.gca()
    ax.set_title('Frame {}'.format(title))
    root_node = df['parent'].min()
    G.add_node(root_node)

    for parent in np.unique(df['parent'].values):
        children = df.loc[df['parent'] == parent]
        for index, child in children.iterrows():
            node = child['child']
            G.add_edge(parent, node)
            G.nodes[node]['size'] = child['child_size']


    pos = graphviz_layout(G, prog='dot')
    size = nx.get_node_attributes(G, 'size')

    labels = {root_node: root_node}
    for k, value in size.items():
        labels[k] = "\n{}\n {} p.\n".format(k, value)
    nx.draw_networkx(G, pos, labels = labels, ax=ax, with_labels=True, node_size=2500, font_size="14", width = 2, node_color='r', font_weight = 'bold')


def plot_results(radar_df, method, cluster_labels, prelabels, boxes, score, frame, scene):

        colormap = {-1: to_rgba('darkgray', alpha=0.1), 0: to_rgba('cornflowerblue'), 1: to_rgba('blue'),
             2: to_rgba('lightcoral'),  3: to_rgba('violet'), 4: to_rgba('orange'), 5: to_rgba('red'),
             6: to_rgba('green'), 7: to_rgba('black'), 8: to_rgba('yellow'),  9: to_rgba('pink'),
             10: to_rgba('violet'), 11: to_rgba('purple'), 12: to_rgba('lightgreen'), 13: to_rgba('coral'),
             14: to_rgba('beige'), 100: to_rgba('cyan'), 101: to_rgba('wheat'), 102: to_rgba('orchid'),
             103: to_rgba('indigo'), 104: to_rgba('cadetblue')}

        true_labels = radar_df['label'].values
        palette = sns.color_palette("hls", cluster_labels.max()+1)
        cluster_colors = list()
        for col in cluster_labels:
            if col < 0:
                cluster_colors.append(to_rgba('darkgray', alpha=0.1))
            elif col > 99:
                cluster_colors.append(colormap[col])
            else:
                cluster_colors.append(palette[col])

        x = radar_df['x'].values
        y = radar_df['y'].values
        xlabel = "Range (radar front) in meters"
        ylabel = "Left/right side of ego vehicle (in meters)"

        if CONFIG['plot_true_labels']:
            fig = plt.figure(figsize=(18,6), dpi=144)
            ax = fig.add_subplot(1,2,1)
        else:
            fig = plt.figure(figsize=(10,6), dpi=144)
            ax = fig.add_subplot()

        for box in boxes:
            box.render(ax)
        ax.set_title("Scene {}/{}: {} clusters extracted by {}".format(scene, frame, cluster_labels.max()+1, method));

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(0, 0, '>', color='red') #ego vehicle
        ax.locator_params(nbins=5)
        ax.scatter(x,y, c=cluster_colors, s=100,  edgecolors='k', zorder=100)

        if CONFIG['plot_true_labels']:

            true_labels = radar_df['label'].values
            ax2 = fig.add_subplot(1,2,2)
            ax2.plot(0, 0, '>', color='red')
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel(ylabel)
            for box in boxes:
                box.render(ax2)

            labels_to_show = true_labels
            ax2.set_title("Radar points w.r.t true labels")
            if method == HDBSCAN_B3F:
                ax2.set_title("Radar points w.r.t prelabels")
                labels_to_show = prelabels

            labels_colors = list()
            for label in labels_to_show:
                if label < 0:
                    labels_colors.append(to_rgba('darkgray',alpha=0.1))
                else:
                    labels_colors.append(colormap[label])
            ax2.scatter(x,y, c=labels_colors, s=100, edgecolors='k')
            ax2.locator_params(nbins=5)

def process_dataset(radar_df, boxes, scene, frame, method, epsilon, seed):

    print("-----------------------------------")
    better_alternative = 0
    coords = list(zip(radar_df['x'], radar_df['y'], radar_df['velocity']))

    velocities = radar_df['velocity'].values
    directions = radar_df['motion'].values
    azimuth = radar_df['x'].values
    distance = radar_df['y'].values
    xy = np.asarray(list(zip(azimuth, distance)))

    prelabels = None
    if method == HDBSCAN_B3F:
        prelabels = get_true_labels_percentage(radar_df, radar_df['label'].values, CONFIG['sample_fraction'], seed)

    cluster_labels, alternatives = cluster_radar_data(coords, prelabels, velocities, directions, xy, epsilon, method, frame)

    evaluated_labels = cluster_labels
    score, radar_df = evaluate(cluster_labels, radar_df)
    evaluated_labels = radar_df['cluster_labels'].values.copy()
    print("Frame score: {:.2f}".format(score))

    if CONFIG['evaluate_alternatives'] and method in (HDBSCAN_CONSTRAINT, HDBSCAN_CE, HDBSCAN_EC):
            original_score = score
            if alternatives is not None and len(alternatives) > 0:
                for labels in alternatives:

                     alt_score, alt_df = evaluate(labels, radar_df)
                     if alt_score > score:
                        print("Found better alternative (score {0:.2f} > {1:.2f})".format(alt_score, score))
                        score = alt_score
                        evaluated_labels = alt_df['cluster_labels'].values.copy()

            if score > original_score:
                print("Chosen result with score: {:.2f}.".format(score))
                better_alternative+=1


    if CONFIG['plot_results']:
        plot_results(radar_df, method, evaluated_labels, prelabels, boxes, score, frame, scene)

    return (score, better_alternative)

def apply_method(method, scene, seed):

    #Figure 1a: 1003/21 Figure 1b: 0239/11 Figure 1c: 0400/13:  Figure 1d: 0553/12
    epsilon_dict = {HDBSCAN_B3F: 0, HDBSCAN_EOM: 0, HDBSCAN_CONSTRAINT: 0, HDBSCAN_EC: 1.5, HDBSCAN_CE: 1.5, DBSCAN_STAR: 4}
    better_alternative_chosen = 0
    scene_results = list()
    single_frame = -1

    sourceDir = os.path.join(basePath,scene);
    sourceDataDir = os.path.join(sourceDir, "data");
    sourceBoxDir = os.path.join(sourceDir, "boxes");

    for index, file in enumerate(sorted(os.listdir(sourceDataDir))):
        file_name = file.split(".")[0]
        frame = int(file_name.split("_")[-1])
        if single_frame != -1 and frame != single_frame:
            continue

        df = pd.read_csv(os.path.join(sourceDataDir, file))
        print("#############################################################################")
        print("############################# SCENE {} FRAME {} ###########################".format(scene,frame))
        print("#############################################################################")

        box_file = sorted(os.listdir(sourceBoxDir))[index]
        boxes = np.load(os.path.join(sourceBoxDir, box_file), allow_pickle=True)

        frame_row = {}
        frame_row["Scene"] = scene
        frame_row['Frame'] = frame
        epsilon = epsilon_dict[method]
        score, better_alternative = process_dataset(df, boxes, scene, frame, method, epsilon, seed)
        better_alternative_chosen+=better_alternative
        frame_row[method] = round(score,2)
        scene_results.append(frame_row)

    if CONFIG['evaluate_alternatives'] and method in (HDBSCAN_CONSTRAINT, HDBSCAN_CE, HDBSCAN_EC):
        print("Better alternatives chosen: {} times.".format(better_alternative_chosen))
    return scene_results


if __name__ == '__main__':

    methods_available = [HDBSCAN_B3F, HDBSCAN_CONSTRAINT, HDBSCAN_CE, HDBSCAN_EC, HDBSCAN_EOM, DBSCAN_STAR]
    scenes_available = ['1003', '0239', '0400', '0553']

    method_to_run = HDBSCAN_EC
    scene_to_run = '0553'

    if method_to_run == HDBSCAN_B3F:
        scene_results = list()
        for seed in range (0, 100):
             results_for_run = apply_method(HDBSCAN_B3F, scene_to_run, seed)
             scene_results.extend(results_for_run)
    else:
        scene_results = apply_method(method_to_run, scene_to_run, 0)

    scene_df = pd.DataFrame(scene_results)
    scene_average = round(scene_df[method_to_run].mean(), 2)
    summary_row = {"Scene": scene_to_run, "Frame": "Scene Average", method_to_run: scene_average}
    scene_df = scene_df.append(pd.DataFrame(summary_row, index=[len(scene_df)]))

    print("#############################################################################")
    if method_to_run == HDBSCAN_B3F:
        prelabeled = CONFIG['sample_fraction']*100
        print("B3F average ARI after 100 runs with {} percent prelabels for scene {}: {}".format(prelabeled, scene_to_run, scene_average))
    else:
        print("Average ARI for scene {} with {}: {}".format(scene_to_run, method_to_run, scene_average))
