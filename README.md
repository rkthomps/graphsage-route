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
- The resulting model will be saved to the directory "models/<current date and type>"

5. Evaluate the GraphSage agent against classical routing algorithms.
- 




