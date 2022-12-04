
"""
Script to use the peeringdb api to create an internet topology
that has the information needed to easily reconstruct the topology
in c++ for quick simulation.

This script assumes the user is in the root directory of the project.
Therefore, it can be invoked by "python3 python_src/create_peering_topology.py."

This script requires zip code data from https://www.listendata.com/2020/11/zip-code-to-latitude-and-longitude.htmlx.
There should be a file with the locations for each continent. These files should be concatenated into one csv

Note that due to the limitations of the free peering db api, sometimes not all of the
data can be gathered at once. 
"""
import sys, os
import requests
import json
import argparse
import pandas as pd
import pdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import tensorflow as tf
import heapq

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

RAW_DATA_LOC = os.path.join("data", "raw")
LOCATION_DATA_LOC = os.path.join("data", "all_places.csv")
TOPOLOGY_DATA_LOC = os.path.join("data", "topology.json")
LINK_DATA_LOC = os.path.join("data", "links.json")
ANALYSIS_LOC = os.path.join("analysis")

DATA_CATEGORIES = ["fac", "ix", "ixfac", "ixlan", "ixpfx",
                   "net", "poc", "netfac", "netixlan", "org",
                   "as_set"]

DEFAULT_LAT_LON = None, None, None

URL_PREFIX = "https://www.peeringdb.com/api/"
    
def gather():
    """
    Get all data from peeringdb
    """
    if not os.path.exists(RAW_DATA_LOC):
        os.makedirs(RAW_DATA_LOC)
    for dc in DATA_CATEGORIES:
        url = URL_PREFIX + dc
        response = requests.get(url)
        write_loc = os.path.join(RAW_DATA_LOC, dc + ".json")
        with open(write_loc, "w") as fout:
            fout.write(json.dumps(
                json.loads(response.content),
                indent=2))


def load_location_data():
    """
    Loads the given location dataset. Provides a country lookup table,
    and a zipcode lookup table. 
    """
    location_data = pd.read_csv(LOCATION_DATA_LOC)
    by_country = location_data.groupby("country code").agg(
        avg_lat=("latitude", "mean"),
        avg_lon=("longitude", "mean"))
    by_country = by_country.dropna()
    country_lookup = dict(zip(by_country.index,
                              zip(by_country["avg_lat"], by_country["avg_lon"])))
    by_zip = location_data.groupby(["country code", "postal code"]).agg(
        avg_lat=("latitude", "mean"),
        avg_lon=("longitude", "mean"))
    by_zip = by_zip.dropna()
    zip_lookup = dict(zip(by_zip.index,
                          zip(by_zip["avg_lat"], by_zip["avg_lon"])))
    return country_lookup, zip_lookup


by_lat = 0
by_zip = 0
by_country = 0
total = 0

def get_lat_lon(org_info, country_lookup, zip_lookup):
    """
    Retrieve latitude and longitude given information about an org

    Args:
        org_info (list[dict]): Information about an org
        country_lookup: Mapping from country code to lat lon
        zip_lookup: Mapping from zip code to lat lon
    """
    global by_lat
    global by_zip
    global by_country
    global total
    total += 1
    lat = org_info["latitude"]
    lon = org_info["longitude"]
    if lat and lon:
        by_lat += 1
        return lat, lon, "by_lat"
    country = org_info["country"]
    if country == "" or country is None:
        return DEFAULT_LAT_LON
    zipcode = org_info["zipcode"]
    country_zip = (country, zipcode)
    if country_zip in zip_lookup:
        by_zip += 1
        zip_lat, zip_lon = zip_lookup[country_zip]
        return zip_lat, zip_lon, "by_zip"
    if country in country_lookup:
        by_country += 1
        country_lat, country_lon = country_lookup[country]
        return country_lat, country_lon, "by_country"
    return DEFAULT_LAT_LON


def calc_haversine(lat1, lon1, lat2, lon2):
    """
    Calculate distance in m as the crow flies given
    latitudes and longitudes
    """
    r = 6371e3 # radius of the earth
    rlat1 = lat1 * np.pi / 180
    rlat2 = lat2 * np.pi / 180
    dlat = (lat2 - lat1) * np.pi / 180
    dlon = (lon2 - lon1) * np.pi / 180

    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(rlat1) * np.cos(rlat2) \
        * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def compute_routing_entry(routing_entry, link):
    new_routing_entry = {
        "d": routing_entry["d"] + link.length,
        "bad": routing_entry["bad"] + link.length,
        "b": routing_entry["b"] + link.bandwidth,
        "bah": routing_entry["bah"] + 1,
        "dab": routing_entry["dab"] + link.bandwidth,
        "h": routing_entry["h"] + 1,
        "weight": 0.0
    }
    return new_routing_entry


def overwrite_routing_entry(old_routing_entry, new_routing_entry):
    # Distance and hops can be checked easily
    if new_routing_entry["d"] < old_routing_entry["d"]:
        old_routing_entry["d"] = new_routing_entry["d"]
    if new_routing_entry["h"] < old_routing_entry["h"]:
        old_routing_entry["h"] = new_routing_entry["h"]

        
    # Check average bandwidth
    if old_routing_entry["bah"] > 0:
        new_band_avg = new_routing_entry["b"] / new_routing_entry["bah"]
        old_band_avg = old_routing_entry["b"] / old_routing_entry["bah"]
        if new_band_avg > old_band_avg:
            old_routing_entry["b"] = new_routing_entry["b"]
            old_routing_entry["bah"] = new_routing_entry["bah"]


    # This is like average bandwidth except weighted by distance
    if old_routing_entry["bad"] == 0:
        return

    if new_routing_entry["bad"] == 0:
        old_routing_entry["bad"] = new_routing_entry["bad"]
        old_routing_entry["dab"] = new_routing_entry["dab"]
        return

    new_ratio = new_routing_entry["dab"] / new_routing_entry["bad"]
    old_ratio = new_routing_entry["dab"] / new_routing_entry["bad"]
    if new_ratio > old_ratio:
        old_routing_entry["bad"] = new_routing_entry["bad"]
        old_routing_entry["dab"] = new_routing_entry["dab"]


class Link:
    def __init__(self, link_id, bandwidth, length=0, utilization=0):
        self.link_id = link_id
        self.bandwidth = bandwidth
        self.utilization = utilization
        self.length = length

    def __lt__(self, other):
        return self.link_id < other.link_id

        
    def get_config(self):
        config = {
            "link_id": self.link_id,
            "bandwidth": self.bandwidth,
            "utilization": self.utilization,
            "length": self.length
        }
        return config

    
    @classmethod
    def from_config(cls, config):
        return cls(**config);
    

class Node:
    def __init__(self, node_id, lat=None, lon=None, method=None,
                 links=None, neighbors=None, routing_table=None):
        self.node_id = node_id
        self.lat = lat
        self.lon = lon
        self.method = method
        self.neighbors = neighbors if neighbors is not None else []
        self.links = links if links is not None else []
        self.neighbor_distances = [] # Only calculated by simulate_bgp
        if routing_table is None:
            routing_table = {
                node_id: {
                    "d": 0.0,      # Distance
                    "bad": 0.0,    # Bandwidth-aware-distance
                    "b": 0.0,      # Bandwidth
                    "bah": 0,      # Bandwidth-aware-hops
                    "dab": 0.0,    # Distance-aware-bandwidth
                    "h": 0,        # Hops
                    "weight": 0.0  # Set by NN
                }}
        self.routing_table = routing_table


    def reset_links(self):
        """
        Set utilization of all links to 0
        """
        for link in self.links:
            link.utilization = 0


    def find_min(self, destination, compare_function):
        """
        Return link weights determined by number of hops. 
        """
        the_min = None
        num_mins = -1
        all_mins = set()
        for i, neighbor in enumerate(self.neighbors):
            if neighbor.node_id == destination:
                the_min = 0
                num_mins = 1
                all_mins = set()
                all_mins.add(i)
                break
            if destination not in neighbor.routing_table:
                continue
            neighbor_hops = compare_function(neighbor.routing_table[destination])
            if the_min is None or neighbor_hops < the_min:
                the_min = neighbor_hops
                num_mins = 1
                all_mins = set()
                all_mins.add(i)
            elif neighbor_hops == the_min:
                num_mins += 1
                all_mins.add(i)

        weights = []
        for i, neighbor in enumerate(self.neighbors):
            if i in all_mins:
                weights.append(1 / num_mins)
            else:
                weights.append(0)
        return weights


    def softmax(self, destination, compare_function):
        """
        Compute the softmax over the logits at each destination
        to get the weights of the outgoing links
        """
        logits = []
        for neighbor in self.neighbors:
            if destination not in neighbor.routing_table:
                logits.append(-1e9)
            logits.append(compare_function(neighbor.routing_table[destination]))
        return tf.nn.softmax(np.array(logits, dtype=np.float32))


    def find_weights(self, method, destination):
        """
        Find all paths to the given destination prune paths
        that have less than prune_path probability
        """
        if method == "hops":
            compare_function = lambda entry: (entry["h"], entry["d"])
        elif method == "distance":
            compare_function = lambda entry: (entry["d"], entry["h"])
        elif method == "bandwidth":
            compare_function = lambda entry: (entry["b"] / entry["bah"], entry["h"])
        elif method == "distance-bandwidth":
            compare_function = lambda entry: (entry["bad"] / entry["dab"], entry["h"])
        elif method == "from-link-logits":
            return self.softmax(destination, lambda entry: entry["weight"])
        elif method == "max-link-logits":
            compare_function = lambda entry: -1 * (entry["weight"])
        return self.find_min(destination, compare_function)
    
            
    def find_paths(self, method, destination, traffic, max_hops, search_beam=2):
        """
        Method indicates the method by which we determine the path
        weights a.k.a the probability a node will send traffic to
        each of its neighbors.

        This has to be beam search or else the algorithm becomes intractable
        """
        terminal_paths = []
        next_candidate_paths = []
        heapq.heappush(next_candidate_paths, (1, [], self)) # prob, traffic, links, node
        hops_away = 0
        while len(next_candidate_paths) > 0:
            candidate_paths = next_candidate_paths
            next_candidate_paths = []
            for (prob, links, node) in candidate_paths:
                next_weights = node.find_weights(method, destination)
                for i, weight in enumerate(next_weights):
                    next_prob = float(weight) * prob
                    node.links[i].utilization += traffic * next_prob
                    new_path = (next_prob,
                                links + [node.links[i]],
                                node.neighbors[i])
                    if new_path[2].node_id == destination:
                        terminal_paths.append(new_path)
                    else:
                        heapq.heappush(next_candidate_paths, new_path)
                        if len(next_candidate_paths) > search_beam:
                            heapq.heappop(next_candidate_paths)
            hops_away += 1
            if hops_away >= max_hops:
                break
        return terminal_paths

        
    def broadcast_to_neighbors(self):
        """
        Broadcast the routing table of this node to its neighbors
        - if neighbor's routing table doesn't have an entry, include it
        - overwrite if neighbor's routing table entry is not better.
        """
        for i, neighbor in enumerate(self.neighbors):
            for k, v in self.routing_table.items():
                new_routing_entry = compute_routing_entry(v, self.links[i])
                if k not in neighbor.routing_table:
                    neighbor.routing_table[k] = new_routing_entry
                    continue
                overwrite_routing_entry(neighbor.routing_table[k], new_routing_entry)

                
    def interpolate(self, iteration):
        """
        Find the average location of neighbors and set
        self location to the average. Return true if
        the location was successfully interpolated. False
        otherwise
        """
        if (self.lat is not None) and (self.lon is not None):
            return True
        lat_sum = 0
        lat_num = 0
        lon_sum = 0
        lon_num = 0
        for neighbor in self.neighbors:
            if (neighbor.lat is None) or (neighbor.lon is None):
                continue
            lat_sum += neighbor.lat
            lat_num += 1
            lon_sum += neighbor.lon
            lon_num += 1
        if (lat_num > 0) and (lon_num > 0):
            self.lat = lat_sum / lat_num
            self.lon = lon_sum / lon_num
            self.method = "interpolated " + str(iteration)
            return True
        return False


    def set_links(self, link_dict):
        new_links = []
        for i, neighbor in enumerate(self.neighbors):
            link_length = calc_haversine(self.lat, self.lon,
                                         neighbor.lat, neighbor.lon)
            new_link = link_dict[self.links[i]]
            new_link.length = link_length
            new_links.append(new_link)
        self.links = new_links


    def fill_neighbors(self, node_dict):
        """
        Replace the node_ids in the neighbors member with references
        to other nodes. 
        """
        new_neighbors = []
        for neighbor in self.neighbors:
            new_neighbors.append(node_dict[neighbor])
        self.neighbors = new_neighbors

    
    def get_config(self):
        """
        Return the json object for this node
        """
        config = {
            "node_id": self.node_id,
            "lat": self.lat,
            "lon": self.lon,
            "method": self.method,
            "neighbors": [n.node_id for n in self.neighbors],
            "links": [l.link_id for l in self.links],
            "routing_table": self.routing_table
        }
        return config


    @classmethod
    def from_config(cls, config):
        """
        Create Node object from the given configuration file.
        note that the node object is not complete after this
        file. All of its members still need to be instantiated
        """
        return cls(**config)


def get_org_lookup_table():
    """
    Get a lookup table for known organization locations
    org_id -> (lat, lon)
    """
    org_lookup = {}
    country_lookup, zip_lookup = load_location_data()
    with open(os.path.join(RAW_DATA_LOC, "org.json")) as fin:
        orgs = json.load(fin)
        for org in orgs["data"]:
            org_lookup[org["id"]] = get_lat_lon(org, country_lookup, zip_lookup)
    return org_lookup


def as_key(asn):
    return "as_" + str(asn)

def ix_key(ix):
    return "ix_" + str(ix)


def create_ases(org_lookup):
    """
    Create AS objects
    """
    as_dict = {}
    with open(os.path.join(RAW_DATA_LOC, "net.json")) as fin:
        ases = json.load(fin)["data"]
        for net in ases:
            if net["org_id"] in org_lookup:
                lat, lon, method = org_lookup[net["org_id"]]
            else:
                lat, lon, method = DEFAULT_LAT_LON
            new_as = Node(as_key(net["asn"]), lat, lon, method)
            as_dict[as_key(net["asn"])] = new_as
    return as_dict


def create_ixes(org_lookup):
    """
    Create IX objects
    """
    ix_dict = {}
    with open(os.path.join(RAW_DATA_LOC, "ix.json")) as fin:
        ixes = json.load(fin)["data"]
        for ix in ixes:
            if ix["org_id"] in org_lookup:
                lat, lon, method = org_lookup[ix["org_id"]]
            else:
                lat, lon, method = DEFAULT_LAT_LON
            new_ix = Node(ix_key(ix["id"]), lat, lon, method)
            ix_dict[ix_key(ix["id"])] = new_ix
    return ix_dict


def create_as_topology(as_dict, ix_dict):
    """
    Add links between ASes and IXes
    """
    links = {} # linkid -> link
    with open(os.path.join(RAW_DATA_LOC, "netixlan.json")) as fin:
        lans = json.load(fin)["data"]
        for lan in lans:
            if lan["speed"] <= 0:
                continue
            links[lan["id"]] = Link(lan["id"], lan["speed"])
            as_id = as_key(lan["asn"])
            ix_id = ix_key(lan["ix_id"])
            as_dict[as_id].neighbors.append(ix_dict[ix_id])
            as_dict[as_id].links.append(lan["id"])
            ix_dict[ix_id].neighbors.append(as_dict[as_id])
            ix_dict[ix_id].links.append(lan["id"])
    return links
    


def propogate_locations(component_dict):
    """
    Interpolate the locations of ASes and IXes with missing
    latitudes and longitudes. Given a single graph where all nodes are guarenteed
    to be reachable
    """
    missing_locations = 1
    iteration = 0
    while missing_locations > 0:
        missing_locations = 0
        total_locations = 0
        iteration += 1
        for node_id, node in component_dict.items():
            missing_locations += 1 if (not node.interpolate(iteration)) else 0
            total_locations += 1
        print(missing_locations, "out of", total_locations, "locations missing after iteration", iteration, "of interpolation")

        
def find_components(as_dict, ix_dict):
    to_explore = set()
    all_nodes = {}
    for asn, as_obj in as_dict.items():
        all_nodes[asn] = as_obj
        to_explore.add(asn)
    for ix_id, ix_obj in ix_dict.items():
        all_nodes[ix_id] = ix_obj
        to_explore.add(ix_id)

    components = []
    while len(to_explore) > 0:
        start_node_id = to_explore.pop()
        new_frontier = [start_node_id]
        component = {start_node_id: all_nodes[start_node_id]}
        while len(new_frontier) > 0:
            frontier = new_frontier
            new_frontier = []
            for node_id in frontier:
                for neighbor in all_nodes[node_id].neighbors:
                    if neighbor.node_id not in to_explore:
                        continue
                    to_explore.remove(neighbor.node_id)
                    new_frontier.append(neighbor.node_id)
                    component[neighbor.node_id] = neighbor
        components.append(component)
    return components


def set_link_lengths(component, links):
    for node_id, node in component.items():
        node.set_links(links)


def plot_topology(component):
    """
    Plot the topology of the given graph of ASes and IXes
    """
    ases = {}
    ixes = {}
    for key, node in component.items():
        if key.startswith("as"):
            ases[key] = node
        elif key.startswith("ix"):
            ixes[key] = node

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    base = world.plot(color="white", edgecolor="black")
    base.set_title(r"Topology Created Using Peering DB")

    for key, node in ases.items():
        for neighbor in node.neighbors:
            base.plot([node.lon, neighbor.lon],
                      [node.lat, neighbor.lat],
                      color="blue",
                      alpha=0.01,
                      linewidth=0.3)

    plt.savefig(os.path.join(ANALYSIS_LOC, "worldplot.png"))


def count_links(nodes):
    """
    Counts the links in the given connected topology
    """
    link_ids = set()
    for node_id, node in nodes.items():
        for link in node.links:
            link_ids.add(link.link_id)
    return len(link_ids)
        


def dump_topology(component, links):
    """
    Write the ases and ixes of the connected component
    """
    configs = [node.get_config() for (key, node) in component.items()]
    with open(TOPOLOGY_DATA_LOC, "w") as fout:
        fout.write(json.dumps(configs, indent=2))
    link_configs = [link.get_config() for (l_id, link) in links.items()]
    with open(LINK_DATA_LOC, "w") as fout:
        fout.write(json.dumps(link_configs, indent=2))


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool to interface with peeringdb to create a topology of the AS-level internet")
    parser.add_argument("-g", "--gather", action="store_true",
                        help="Gather the data from peeringdb. Copy it locally.")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Produce a global plot of the produced topology")
    args = parser.parse_args(sys.argv[1:])
    if args.gather:
        gather()
        
    org_lookup = get_org_lookup_table()
    
    print("By Lat", by_lat)
    print("By Zip", by_zip)
    print("By Country", by_country)
    print("Total", total)

    print("\nCreating ASes")
    as_dict = create_ases(org_lookup)
    print("\nCreating IXes")
    ix_dict = create_ixes(org_lookup)
    print("\nCreating Topology")    
    links = create_as_topology(as_dict, ix_dict)

    print("Created an Internet Topology with",
          len(as_dict) + len(ix_dict),
          "nodes and", len(links), "edges")

    print("\nFinding Components...")
    components = find_components(as_dict, ix_dict)
    largest_component = max(components, key=lambda x: len(x))
    print("\nInterpolating Locations")
    propogate_locations(largest_component)


    if args.plot:
        print("\nPlotting Topology")
        plot_topology(largest_component)

    print("\n Setting Link Lengths")
    set_link_lengths(largest_component, links)

    print("Ended with an Internet Topology with",
          len(largest_component), "nodes and",
          count_links(largest_component), "edges")

    print("\n Dumping Topology")
    dump_topology(largest_component, links)
    print("Topologies Written")

    
    



