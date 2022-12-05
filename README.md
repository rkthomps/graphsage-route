# graphsage-route
Implementation of a reinforcement learning agent that minimizes the delay of routing over a realistic AS-level topology. 

*Author*: Kyle Thompson (rkthomps@calpoly.edu)

*Project*: CSC-564 (Research Topics in Computer Networks) final project.

*Professor*: Professor Devkishen Sisodia

# Running the Source Code
1. In the data folder, you will find two zipped files. 
  - The file "data/110522_peering_db.tar.gz" contains data from the [Peering DB API](https://www.peeringdb.com/apidocs/).
  - The file "data/all_places_comp.csv.gz" contains the latitudes and longitudes of global zip codes. 
  - Before running any scripts, you must unzip the two files above.
  - Untaring the "data/110522_peering_db.tar.gz" file should give you a "data/raw/*.json" directory structure
  - Unzipping the "data/all_places_comp.csv.gz" will give you a "data/all_places_comp.csv" file. 
  You will need to either move this file to "data/all_places.csv" or change `LOCATION_DATA_LOC` variable in "python_src/create_peering_topology.py"
 
2. You need to create a virtual environment.
- First run `python3 -m venv venv` to create the environment. 
- Then run `pip3 install -r requirements.txt` to install the necessary packages.
- run `source venv/bin/activate` to activate the virtual environment

3. Creating a realistic peering topology.
- run `python3 python_src/create_peering_topology.py` to create a real-world AS-level topology. 
The resulting topology is saved in the "data/topology.json" and "data/links.json" files.

4. Train a GraphSage agent.
- run `python3 python_src/train_sage.py` to train a graphsage agent.
- You can modify the training hyperparameters by modifying the file "python_src/train_sage.py".
- The resulting model will be saved to the directory "models/\<current date and time\>"

5. Evaluate the GraphSage agent against classical routing algorithms.
- run `python3 python_src/simulate_bgp.py -h` to see the available arguments. 
- The command to get the final results from the paper was `python3 python_src/simulate_bgp.py -m models/20221202_123431 3 20 "hops;distance;bandwidth;distance-bandwidth" "0.2;0.4;0.6;0.8;1" "16;64;256;1024;8192"`
- The results will be saved to the directory "results/\<current date and time\>"

# Source Code Description
- "python_src/create_peering_topology.py"
  - Definition of a Node in the topology
  - Definition of a Link in the topology
  - Logic for constructing an AS-level topology
  - Logic for finding paths between two nodes given a metric to route on.
- "python_src/simulate_bgp.py"
  - Logic for broadcasting routing tables over the topology.
  - Logic for simulating traffic over subsets of the generated AS-level topology. 
  - Can simulate traffic using a GraphSage model to assign weights, or using a classical routing method.
  - Saves the result to the "results" directory along with plots for analysis. 
- "python_src/graph_sage.py"
  - Implementation of GraphSage in tensorflow.
  - Implementation of on training step of ES in tensorflow.
- "python_src/train_sage.py"
  - Wrapper module for the training step defined in "graph_sage.py."
  Instantiates a GraphSage model and trains it for a number of training steps using the hyperparameters
  defined in the module. 
  Saves the resulting model to a directory where it can later be loaded and used for evaluation. 




