# Constraint-based-Clustering

This repository provides code as described in our paper [Constraint-Based Hierarchical Cluster Selection in Automotive Radar Data](https://www.mdpi.com/1424-8220/21/10/3410). It integrates cluster-level constraints into the hierarchical clustering algorithm HDBSCAN.


The application of a distance threshold (see our paper [A Hybrid Approach To Hierarchical Density-based Cluster Selection](https://ieeexplore.ieee.org/document/9235263)) is already integrated into the existing [Python implementation by McInnes et al.](https://github.com/scikit-learn-contrib/hdbscan), see the [documentation](https://hdbscan.readthedocs.io/en/latest/how_to_use_epsilon.html).

The code in this repository is based on the same HDBSCAN implementation. The most important modifications were made in `_hdbscan_tree.pyx`. In addition, we modified `hdbscan_.py` in order to allow different parameters and return values.

Please note that the uploaded code includes hard-coded values for radar data constraints as described in our paper. Further below, you will find some tips how to create a customized constraint-based HDBSCAN version.

The file `hdbscan_constraint_radar.py` runs our HDBSCAN version for four labelled nuscenes traffic scenes (`nuscenes_data_labeled`) as described in the paper. To run a different scene or change the cluster selection method, you need to adjust the parameters in `__main__` accordingly. Further, the `CONFIG` object helps to adjust some configurations, such as plotting results. 


----------
Prerequisites
----------

    pip install -r requirements.txt

Aside from packages for the actual HDBSCAN installation, `requirements.txt` lists some packages needed for running `hdbscan_constraint_radar.py`. If you want to run `hdbscan_constraint_radar.py` with visualization of condensed hierarchy trees (see `CONFIG` options), please note that you need to [install PyGraphviz](https://pygraphviz.github.io/documentation/stable/install.html) separately. 

----------
Installation
----------
The HDBSCAN setup file from the [original repository](https://github.com/scikit-learn-contrib/hdbscan) is used with modified directory paths.

    python setup.py build_ext --inplace

In case you want to create your own constraint-based version and make changes to `_hdbscan_tree.pyx` in the `hdbscan_constraint` folder, make sure to re-run this command afterwards.

----------
Customization
----------
In our constraint-based version, HDBSCAN is used as follows:

```python
import hdbscan_constraint
[...]
clusterer = hdbscan_constraint.HDBSCAN(min_cluster_size=minPts,cluster_selection_epsilon=epsilon, cluster_selection_method='constraint+e', allow_single_cluster=allow_single_cluster, prelabels=prelabels, velocities=velocities, xy=xy, directions=directions) 
cluster_labels, alternatives = clusterer.fit_predict(data)
```

Compared to the original HDBSCAN, we pass additional parameters (_prelabels_, _velocities_, _xy_, _directions_) and also use custom selection method names (in this case, *constraint+e*). All of this can be modified in the `hdbscan_.py` file according to your needs. For example, instead of an array of velocity values you might want to pass an array with values of some other type that can then be assigned to the cluster candidates within the condended tree and later used to decide about the selection of clusters.

In the current implemention, we pass our parameters directly to the `condense_tree` function from `_hdbscan_tree` (see `_hdbscan_tree.pyx`). We modified this function such that it returns a tuple with both the tree structure and the constraints as a separate list.

Several variations of this approach are possible. For example, instead of computing constraints directly during creation of the condensed cluster tree, you could compute them separately, similarly as it is done by the functions `compute_stability` and `compute_b3f_measure`. 

You might also want to introduce additional parameters to pass threshold values, instead of hard-coding them into `_hdbscan_tree`. Have a look at `_tree_to_labels` in `hdbscan_.py` to see how values are passed to and from `_hdbscan_tree`. In our case, we pass the list of constraints we obtained from `condense_tree` to `get_clusters`, where the different cluster selection methods are applied. We modified the return values of this function such that alternative labels can later be returned by `fit_predict`  in `hdbscan_.py`.
