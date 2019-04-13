use std::collections::{HashSet, HashMap};
use crate::numerics::*;

/**
 * Defines what we merge against what
 */
#[derive(Debug)]
pub enum Merge {
    Sequence2Sequence,
    Sequence2Cluster,
    Cluster2Sequence,
    Cluster2Cluster,
}

/**
 * Defines what which instances we merge into which cluster
 */
#[derive(Debug)]
pub struct ClusteringOperation {
    pub merge_i: usize,
    pub merge_j: usize,
    pub into: usize,
    distance: f32,
    pub operation: Merge,
}

/**
 * Performs hierarchical clustering.
 * Holds temporary data during dendogram construction.
 */
pub struct AgglomerativeClustering {
    /// Parent pointers similar to the union find data structure.
    parents: Vec<usize>,
    distances: Vec<f32>,
    n_instances: usize,
    n_clusters:  usize,
}

impl AgglomerativeClustering {
    pub fn cluster_sets(operations: &[ClusteringOperation], cluster_ids: &HashSet<usize>, n_instances: usize) -> Vec<Vec<usize>> {
        let mut results: HashMap<usize, Vec<usize>> = HashMap::new();
        for op in operations {
            let i = op.merge_i;
            let j = op.merge_j;
            let k = op.into;
            let mut cluster = vec![];   
            if let Some(c) = results.get(&i) { 
                cluster.extend(c);
            } else {
                cluster.push(i);
            }
            if let Some(c) = results.get(&j) { 
                cluster.extend(c);
            } else {
                cluster.push(j);
            }
            results.insert(k, cluster);
        }
        let mut grouped = vec![];
        for cluster in cluster_ids.iter() {
            match results.get(cluster) {
                Some(result) => grouped.push(result
                    .iter()
                    .filter_map(|i| if *i < n_instances { Some(*i) } else { None })
                .collect()),
                _ => println!("Cluster not found: {} | Singular cluster", cluster)
            }
        }
        grouped
    }

    /**
     * Initialise agglomerative clustering setting each instance as it's own cluster
     */
    pub fn clustering(distances: Vec<f32>, n_instances: usize, perc: f32) -> (Vec<ClusteringOperation>, HashSet<usize>) {
        let n_clusters = n_instances;
        let mut parents = vec![];
        for i in 0..n_instances {
            parents.push(i);
        }        
        let mut dendogram = AgglomerativeClustering {
            parents,
            distances: distances.clone(),
            n_instances,
            n_clusters,
        };
        let mut cluster_result = vec![];
        let threshold = percentile(&mut distances.clone(), perc);
        println!("Clustering with {}", threshold);
        let mut distance = 0.0;
        while dendogram.n_clusters > 1 && distance < threshold {            
            let operation = dendogram.merge();
            distance = operation.distance;
            cluster_result.push(operation);        
        }
        (cluster_result, dendogram.clusters())
    }

    /**
     *  Find the cluster assignment for an instance
     */
    fn cluster(&self, i: usize) -> usize {
        let mut p = i;
        while p != self.parents[p] {
            p = self.parents[p];
        }
        p
    }

    /**
     * Compute clustrer assignment for each instance
     */
    fn assignment(&self) -> Vec<usize> {
        (0..self.n_instances).map(|i| self.cluster(i)).collect()
    }

    /**
     * Merge two clusters by adding a new node with the
     * two clusters as a child node
     */
    fn merge_clusters(&mut self, p: usize, q: usize) -> usize {
        let k = self.parents.len();
        self.parents[p] = k;
        self.parents[q] = k;
        self.parents.push(k);
        self.n_clusters -= 1;
        k
    }

    /**
     * Compute the set of top level clusters
     */
    fn clusters(&self) -> HashSet<usize> {
        (0..self.n_instances).map(|i| self.cluster(i)).collect()
    }

    /**
     * Average linkage between instance i and j
     */
    fn linkage(&self, assignment: &[usize], i: usize, j: usize) -> f32 {
        let mut size_x = 0.0;
        let mut size_y = 0.0;
        let mut distance = 0.0;
        for x in 0..assignment.len() {
            if assignment[x] == i {
                size_y = 0.0;
                for y in 0..assignment.len() {
                    if assignment[y] == j {
                        distance += self.distances[x * self.n_instances + y];
                        size_y += 1.0;
                    }
                }
                size_x += 1.0;
            }
        }
        distance / (size_x * size_y)
    }

    /**
     * Merges the best two instances under complete linkage, returns merge operation
     */
    pub fn merge(&mut self) -> ClusteringOperation {
        let assignment = self.assignment();
        let clusters = &self.clusters();
        let mut min_linkage = std::f32::INFINITY;
        let mut min_merge: (usize, usize) = (0, 0);
        for target_i in clusters {
            for target_j in clusters {
                if target_i != target_j {
                    let linkage = self.linkage(&assignment, *target_i, *target_j);
                    if linkage < min_linkage {
                        min_linkage = linkage;
                        min_merge = (*target_i, *target_j);
                    }
                }
            }
        }
        let (p, q) = min_merge;
        let k = self.merge_clusters(p, q);
        let op = if p < self.n_instances && q < self.n_instances {
            Merge::Sequence2Sequence
        } else if p >= self.n_instances && q >= self.n_instances {
            Merge::Cluster2Cluster
        } else if p >= self.n_instances && q < self.n_instances {
            Merge::Cluster2Sequence
        } else {
            Merge::Sequence2Cluster
        };
        ClusteringOperation {
            merge_i: p,
            merge_j: q,
            into: k,
            distance: min_linkage,
            operation: op,
        }
    }
}
