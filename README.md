# Mean-Field Approximation with Perturbation for Steady-State Computation

## Project Description
This repository implements the **Mean-Field Approximation (MFA) enhanced by perturbation** for efficient equilibrium state estimation. MFA offers a fast, decoupled approximation, while the perturbation correction refines the estimate in just a few iterations. Leveraging MFA’s high-quality initial conditions and decoupling strategy, the method scales to networks with billions of interactions on standard CPUs. This approach enables the efficient analysis of large-scale complex systems.

## Dynamics 
We consider four types of dynamical systems with the following general form:

\[
\frac{dx_i}{dt} = F(x_i) + \sum_j A_{ij} G(x_i, x_j)
\]

where:
- \( F(x_i) \) governs the intrinsic behavior of node \( i \).
- \( G(x_i, x_j) \) describes interactions between nodes.
- \( A_{ij} \) represents the adjacency matrix of the network.

 
1. **Mutualistic Dynamics**: Models species abundance with logistic growth and mutualistic interactions.
   \[
   F(x_i) = B + x_{i} \left(1- \frac{x_{i}}{K}\right) \left( \frac{x_{i}}{C} -1 \right), \quad G(x_i, x_j) = \frac{x_{i}x_{j}}{D + E x_{i} + H x_{j}}
   \]
2. **Regulatory Dynamics**: Describes gene regulation using Michaelis-Menten kinetics.
   \[
   F(x_i) = -B x_i^f, \quad G(x_i, x_j) = \frac{x_j^h}{x_j^h + 1}
   \]
3. **Epidemic Dynamics**: Follows an SIS model for disease spread.
   \[
   F(x_i) = -x_i, \quad G(x_i, x_j) = B(1 - x_i)x_j
   \]
4. **Neuronal Dynamics**: Models neural activation using a sigmoid function.
   \[
   F(x_i) = -x_i, \quad G(x_i, x_j) = \frac{1}{1 + e^{-\tau(x_j - \mu)}}
   \]

## Topology  
We conduct experiments on two types of synthetic network topologies:
1. **Erdős-Rényi (ER) Networks**: Random graphs where each edge exists independently with probability \( p \). The degree distribution follows a Poisson distribution.
2. **Scale-Free (SF) Networks**: Networks with a power-law degree distribution, where a few nodes have significantly higher degrees than others.

## How to Run Experiments
To run an experiment, navigate to the `run_scripts` directory and execute:
```sh
./{experiment}.sh
```
Replace `{experiment}` with one of the following:
- `er_density`: Varying density in ER networks.
- `sf_heterogeneity`: Changing degree heterogeneity in SF networks.
   - We provide pre-generated scale-free networks, stored in `/sf_networks` with varying heterogeneity. 
   - The functionality to generate new graphs with varying heterogeneity is given in function `generate_networks()` in `sf_heterogeneity.py`.
- `network_size`: Scaling experiments with different network sizes.
- `huge_net`: Running MFA-Perturbation on large-scale networks.

  
## How to Get Data
### Synthetic Data
Specify arguments when running the experiments:
- `dynamics`: Choose from `eco` (Mutualistic), `gene` (Regulatory), `epi` (Epidemic), or `wc` (Neuronal).
- `topo`: Select `er` or `sf`.
- `n`: Set network size.
- `k`: Define the average degree.

### Real-World Data
For real networks, download datasets using:
```sh
wget {snap_dataset_url} -P data/
```
Replace `{snap_dataset_url}` with the appropriate dataset link. Store the downloaded data in the `data/` directory.
 
We provide preprocessing for the eight datasets adopted in the manuscript:
Email-EU, Amazon0302, Pokec, Friendster, PP-Decagon, GG-central_nervous_system, PP-Pathways, CC-Neuron.

