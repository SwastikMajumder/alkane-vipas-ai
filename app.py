import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import itertools
import matplotlib.pyplot as plt
import copy
import random
from collections import defaultdict, deque
import math
import cv2
import secrets

class SC:
    @staticmethod
    def choice(sequence):
        """Return a random element from the non-empty sequence."""
        if not sequence:
            raise ValueError("Cannot choose from an empty sequence.")
        index = secrets.randbelow(len(sequence))
        return sequence[index]

    @staticmethod
    def shuffle(sequence):
        """Shuffle the sequence in place."""
        if not sequence:
            return  # Return the empty sequence as is
        shuffled = sequence[:]
        n = len(shuffled)
        for i in range(n):
            j = secrets.randbelow(n - i) + i  # Pick a random index to swap with
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]  # Swap the elements
        return shuffled

    @staticmethod
    def randint(a, b):
        """Return a random integer N such that a <= N <= b."""
        return a + secrets.randbelow(b - a + 1)

    @staticmethod
    def uniform(a, b):
        """Return a random float N such that a <= N < b."""
        return a + (b - a) * secrets.randbelow(1000000) / 1000000.0
      
def crop_background(pil_image):
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    binary_inv = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:

        x, y, w, h = cv2.boundingRect(contours[0])

        cropped_image = image[y:y+h, x:x+w]

        cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        return cropped_pil_image
    else:
        return pil_image  
class Graph:
    def __init__(self):
        self.nodes = []  
        self.edges = []  
    def add_node(self, node_id, value="C"):
        if node_id not in [node[0] for node in self.nodes]:
            self.nodes.append([node_id, value])
    def edge_exists(self, node1_id, node2_id):
        return any(edge for edge, _ in self.edges if set(edge) == {node1_id, node2_id})
    def add_edge(self, node1_id, node2_id, value=1):
        self.edges.append([[node1_id, node2_id], value])

    def build_adjacency_list(self):
        adj_list = {node[0]: [] for node in self.nodes}
        for edge, value in self.edges:
            node1_id, node2_id = edge
            adj_list[node1_id].append(node2_id)
            adj_list[node2_id].append(node1_id)  
        return adj_list
    def degree(self, node_id):
        adj_list = self.build_adjacency_list()
        return len(adj_list.get(node_id, []))
    def find_all_cycles(self):
        adj_list = self.build_adjacency_list()
        visited = set()
        recStack = []
        all_cycles = []

        def dfs(v, parent):
            visited.add(v)
            recStack.append(v)

            for neighbor in adj_list[v]:
                if neighbor not in visited:
                    if dfs(neighbor, v):
                        return True
                elif neighbor != parent and neighbor in recStack:

                    cycle_start_index = recStack.index(neighbor)
                    cycle = recStack[cycle_start_index:] + [neighbor]
                    all_cycles.append(cycle)

            recStack.pop()
            return False

        for node_id, _ in self.nodes:
            if node_id not in visited:
                dfs(node_id, None)

        if len(all_cycles) == 0:
            return []  
        elif len(all_cycles) == 1:
            return list(set(all_cycles[0]))  
        else:
            return None  
    def get_edges(self, node_id):
        """Returns the edges connected to a specific node_id."""
        return [edge for edge in self.edges if node_id in edge[0]]
    def add_hydrogens(self):
        """Automatically add hydrogen atoms to carbon atoms to ensure each carbon has four bonds."""
        hydrogens = []  
        count_hydrogen = len(self.nodes)+1
        for node_id, value in self.nodes:
            if value == "C":

                existing_bonds = sum([edge[1] for edge in self.get_edges(node_id)])
                needed_hydrogens = 4 - existing_bonds

                for _ in range(needed_hydrogens):
                    hydrogen_id = count_hydrogen
                    count_hydrogen += 1
                    self.add_node(hydrogen_id, value="H")
                    self.add_edge(node_id, hydrogen_id)
                    hydrogens.append(hydrogen_id)

        return hydrogens
    def fruchterman_reingold(self, iterations=1000, width=400, height=400):

      positions = {node[0]: (SC.uniform(0, width), SC.uniform(0, height)) for node in self.nodes}

      for _ in range(iterations):
          forces = {node_id: [0, 0] for node_id in positions.keys()}

          for node1 in positions:
              for node2 in positions:
                  if node1 != node2:
                      dx = positions[node1][0] - positions[node2][0]
                      dy = positions[node1][1] - positions[node2][1]
                      distance = math.sqrt(dx ** 2 + dy ** 2) + 1e-10  
                      repulsion_force = 1000 / distance**2  
                      forces[node1][0] += dx / distance * repulsion_force
                      forces[node1][1] += dy / distance * repulsion_force

          for (node1_id, node2_id), value in self.edges:
              if node1_id in positions and node2_id in positions:
                  dx = positions[node1_id][0] - positions[node2_id][0]
                  dy = positions[node1_id][1] - positions[node2_id][1]
                  distance = math.sqrt(dx ** 2 + dy ** 2) + 1e-10
                  attraction_force = distance ** 2 / 100  
                  forces[node1_id][0] -= dx / distance * attraction_force
                  forces[node1_id][1] -= dy / distance * attraction_force
                  forces[node2_id][0] += dx / distance * attraction_force
                  forces[node2_id][1] += dy / distance * attraction_force

          for node_id in positions:
              dx, dy = forces[node_id]
              positions[node_id] = (positions[node_id][0] + dx * 0.1, positions[node_id][1] + dy * 0.1)

              positions[node_id] = (
                  max(20, min(positions[node_id][0], width - 20)),
                  max(20, min(positions[node_id][1], height - 20))
              )

      return positions

    def draw_graph(self, width=500, height=500):

      image = Image.new('RGB', (width, height), 'white')
      draw = ImageDraw.Draw(image)

      positions = self.fruchterman_reingold(width=width, height=height)

      for node_id, value in self.nodes:
          x, y = positions[node_id]
          draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill='lightblue', outline='black')

      for (node1_id, node2_id), value in self.edges:
          if node1_id in positions and node2_id in positions:
              x1, y1 = positions[node1_id]
              x2, y2 = positions[node2_id]
              draw.line([x1, y1, x2, y2], fill='black', width=2)  

              mid_x = (x1 + x2) / 2
              mid_y = (y1 + y2) / 2

      image = crop_background(image)
      return image

def terminal_carbon(graph):
    output = []
    for node in graph.nodes:
        node_id = node[0]

        adjacent_count = sum(1 for edge in graph.edges if node_id in edge[0])

        if adjacent_count == 1:
            output.append(node_id)

    return output

def all_chain_all(graph, start_node):
    if len(graph.nodes) == 1:
        return [[[graph.nodes[0][0]], []]]
    all_chain_output = []
    sub_chain_output = []
    def all_chain(chain, compound, sub_chain):
        end_reached = True
        atom_iter = []
        for atom in compound.nodes:
            atom_id = atom[0]
            if atom_id not in chain and {atom_id, chain[-1]} in [set(item[0]) for item in compound.edges]:
                atom_iter.append(atom_id)
        if len(atom_iter) == 0:
            all_chain_output.append(chain)
            sub_chain_output.append(sub_chain)
        elif len(atom_iter) == 1:
            atom_id = atom_iter[0]
            all_chain(chain + [atom_id], compound, sub_chain)
        else:
            for atom_id in atom_iter:
                all_chain(chain + [atom_id], compound, sub_chain + [len(chain)]*(len(atom_iter)-1))
    if start_node is None:
        for atom in terminal_carbon(graph):
            all_chain([atom], graph, [])
    else:
        all_chain([start_node], graph, [])
    return [list(x) for x in zip(all_chain_output, sub_chain_output)]
count_hydrogen = 0
def add_hydrogens(graph):
    global count_hydrogen
    hydrogens = []  
    for node in graph.nodes:
        node_id, value = node
        if value == "C":

            existing_bonds = sum([edge[1] for edge in graph.edges if node_id in edge[0]])
            needed_hydrogens = 4 - existing_bonds

            for _ in range(needed_hydrogens):
                hydrogen_id = count_hydrogen
                count_hydrogen += 1
                graph.add_node(hydrogen_id, value="H")
                graph.add_edge(node_id, hydrogen_id)
                hydrogens.append(hydrogen_id)
    return hydrogens

import numpy as np
import matplotlib.pyplot as plt

def fruchterman_reingold(graph, iterations=100, width=800, height=600, k=None):
    """
    Fruchterman-Reingold force-directed graph drawing algorithm.
    :param graph: Input graph with nodes and edges.
    :param width: Width of the output space.
    :param height: Height of the output space.
    :param k: Optimal distance between nodes (optional, defaults to sqrt(area/n)).
    :return: Dictionary of node positions.
    """

    num_nodes = len(graph.nodes)

    positions = {node: np.SC.rand(2) * [width, height] for node in graph.nodes}

    if k is None:
        area = width * height
        k = np.sqrt(area / num_nodes)

    def repulsive_force(d, k):
        return k**2 / d

    def attractive_force(d, k):
        return d**2 / k

    for _ in range(iterations):
        displacements = {node: np.zeros(2) for node in graph.nodes}

        for i, node_i in enumerate(graph.nodes):
            for j, node_j in enumerate(graph.nodes):
                if i != j:
                    delta = positions[node_i] - positions[node_j]
                    dist = np.linalg.norm(delta)
                    if dist > 0:
                        repulsive = repulsive_force(dist, k)
                        displacements[node_i] += (delta / dist) * repulsive

        for edge in graph.edges:
            node1, node2 = edge
            delta = positions[node1] - positions[node2]
            dist = np.linalg.norm(delta)
            if dist > 0:
                attractive = attractive_force(dist, k)
                displacements[node1] -= (delta / dist) * attractive
                displacements[node2] += (delta / dist) * attractive

        for node in graph.nodes:
            positions[node] += displacements[node]

            positions[node] = np.clip(positions[node], 0, [width, height])

    return positions

def draw_graph(graph, positions):
    """
    Draw the graph using the calculated positions.
    :param graph: Graph with nodes and edges.
    :param positions: Dictionary of node positions.
    """
    plt.figure(figsize=(10, 8))

    for edge in graph.edges:
        node1, node2 = edge
        pos1 = positions[node1]
        pos2 = positions[node2]
        plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', lw=2, color="gray")

    for node, pos in positions.items():
        plt.scatter(pos[0], pos[1], s=200, c='lightblue', edgecolors='black', zorder=2)
        plt.text(pos[0], pos[1], str(node), fontsize=12, ha='center', va='center', zorder=3)

    plt.title("Graph Visualization (Fruchterman-Reingold Algorithm)")
    plt.xlim(0, 800)
    plt.ylim(0, 600)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def explorable_subgraph(graph, start_node, forbidden_nodes):

    subgraph = Graph()

    if start_node in [node[0] for node in graph.nodes]:
        visited = set()
        queue = [start_node]

        while queue:
            node = queue.pop(0)
            if node not in visited and node not in forbidden_nodes:
                visited.add(node)
                subgraph.add_node(node, value=[n[1] for n in graph.nodes if n[0] == node][0])

                for edge in graph.edges:
                    (n1, n2), value = edge
                    if n1 == node and n2 not in visited and n2 not in forbidden_nodes:
                        queue.append(n2)
                        if n2 not in forbidden_nodes:
                            subgraph.add_node(n2, value=[n[1] for n in graph.nodes if n[0] == n2][0])
                            subgraph.add_edge(n1, n2, value)
                    elif n2 == node and n1 not in visited and n1 not in forbidden_nodes:
                        queue.append(n1)
                        if n1 not in forbidden_nodes:
                            subgraph.add_node(n1, value=[n[1] for n in graph.nodes if n[0] == n1][0])
                            subgraph.add_edge(n1, n2, value)

    return subgraph

graph = Graph()

graph.add_node(1, "C")
graph.add_node(2, "C")
graph.add_node(3, "C")
graph.add_node(4, "C")

name = ["meth", "eth", "prop", "but", "pent", "hex", "hept", "oct", "non", "dec",\
            "undec", "dodec", "tridec", "tetradec", "pentadec", "hexadec", "heptadec",\
            "octadec", "nonadec", "icos"]
name =[
    "meth", "eth", "prop", "but", "pent", "hex", "hept", "oct", "non", "dec",
    "undec", "dodec", "tridec", "tetradec", "pentadec", "hexadec", "heptadec",
    "octadec", "nonadec", "icos", "henicos", "docos", "tricos", "tetracos",
    "pentacos", "hexacos", "heptacos", "octacos", "nonacos", "triacont",
    "tetracont", "pentacont", "hexacont", "heptacont", "octacont", "nonacont",
    "hect", "dodecacos", "tricosacont", "tetracontad", "pentacontad"
]

name_2 = [
    "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "nona", "deca", 
    "undeca", "dodeca", "trideca", "tetradeca", "pentadeca", "hexadeca", 
    "heptadeca", "octadeca", "nonadeca", "icosa", "henicosa", "docosa", 
    "tricosa", "tetracosa", "pentacosa", "hexacosa", "heptacosa", 
    "octacosa", "nonacosa", "triaconta", "tetraconta", "pentaconta", 
    "hexaconta", "heptaconta", "octaconta", "nonaconta"
]

def neighbor(graph, node):
    output = []
    for item in graph.edges:
        if node == item[0][0]:
            output.append(item[0][1])
        elif node == item[0][1]:
            output.append(item[0][0])
    return list(set(output))

def process(graph, depth=0, suffix=0, start_node=None):
    global name
    global bond_info
    global buffer
    global name_2
    if graph.nodes == []:
        return

    all_chain_output = all_chain_all(graph, start_node)

    def double_bond(graph, chain):
        double = []
        triple = []
        i = 0
        while i < len(chain[0]):
            for edge in graph.edges:
                if chain[0][i] in edge[0]:
                    if edge[1] == 2:
                        double.append(i+1)
                        i = i + 1
                        break
            i = i + 1
        i = 0
        while i < len(chain[0]):
            for edge in graph.edges:
                if chain[0][i] in edge[0]:
                    if edge[1] == 3:
                        triple.append(i+1)
                        i = i + 1
                        break
            i = i + 1
        return chain + [double] + [triple]

    all_chain_output = [double_bond(graph, x) for x in all_chain_output]

    for i in range(len(all_chain_output)):
        all_chain_output[i] += [0]

    all_chain_new = []

    if graph.find_all_cycles() != []:
        cycle = graph.find_all_cycles()
        for i in range(len(cycle)):
            new_cycle = [cycle[(i+j)%len(cycle)] for j in range(len(cycle))]
            subchain = []
            for index, node in enumerate(new_cycle):
                subchain += [index+1]*(len(neighbor(graph, node))-2)
            all_chain_output.append(double_bond(graph, [new_cycle, subchain])+[1])

    max_double = max([len(x[2])+len(x[3]) for x in all_chain_output])
    if max_double > 0:
        all_chain_output = [x for x in all_chain_output if max_double == len(x[2])+len(x[3])]
        least_sum = min([sum(x[2])+sum(x[3]) for x in all_chain_output])
        all_chain_output = [x for x in all_chain_output if least_sum == sum(x[2])+sum(x[3])]

    max_cycle = max([x[4] for x in all_chain_output])
    all_chain_output = [x for x in all_chain_output if max_cycle == x[4]]

    max_len = max([len(x[0]) for x in all_chain_output])

    all_chain_output = [x for x in all_chain_output if max_len == len(x[0])]
    all_chain_output = sorted(all_chain_output, key=lambda x: sum(x[1]))
    all_chain_output = sorted(all_chain_output, key=lambda x: sum(x[2]))[0]

    sub_chain = []
    for i in list(set(all_chain_output[1])):
        count = all_chain_output[1].count(i)
        sel_edge = []
        for item in [x[0] for x in graph.edges]:
            if all_chain_output[0][i-1] == item[0]:
                sel_edge.append(item[1])
            elif all_chain_output[0][i-1] == item[1]:
                sel_edge.append(item[0])
        sel_edge = list(set(sel_edge) - set(all_chain_output[0]))

        for j in range(count):
            forbidden = list(set(all_chain_output[0] + sel_edge) - set([sel_edge[j]]))

            sub_chain.append([str(i), process(explorable_subgraph(graph, sel_edge[j], forbidden), depth+1, i, sel_edge[j])])
    output = []
    sub_chain_type = {}
    for item in sub_chain:
        if item[1] in sub_chain_type.keys():
            sub_chain_type[item[1]].append(item[0])
        else:
            sub_chain_type[item[1]] = [item[0]]
    for key in sorted(sub_chain_type.keys()):
        if len(sub_chain_type[key]) > 1:
            output.append(",".join(sub_chain_type[key]) + "-" + name_2[len(sub_chain_type[key])-2] + "(" + key + ")")
        else:
            output.append(",".join(sub_chain_type[key]) + "-(" + key + ")")
    name_list = [name[max_len-1]]
    if all_chain_output[4] == 1:
        name_list = ["cyclo"]+name_list
    if depth == 0:
        if len(all_chain_output[2])!=0 and len(all_chain_output[3])!=0:
            arr = ["en", "yne"]
        else:
            arr = ["en", "yne"]
        is_bond = False
        for t, item in enumerate(all_chain_output[2:4]):
            if len(item) == 1:
                name_list.append("-" + ",".join([str(x) for x in item]) + "-" + arr[t])
                is_bond = True
            elif len(item) > 1:
                name_list.append("-" + ",".join([str(x) for x in item]) + "-" + name_2[len(item)-2] + arr[t])
                is_bond = True
        if is_bond is False:
            name_list.append("ane")
    else:
        is_bond = False
        if is_bond is False:
            name_list.append("yl")
    return "-".join(output) + "".join(name_list)

def is_connected(edge_list):
    if not edge_list:
        return True  

    graph = defaultdict(list)
    nodes = set()

    for u, v in edge_list:
        graph[u].append(v)
        graph[v].append(u)
        nodes.add(u)
        nodes.add(v)

    start_node = next(iter(nodes))
    visited = set()

    def bfs(node):
        queue = deque([node])
        visited.add(node)

        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    bfs(start_node)

    return len(visited) == len(nodes)

def is_tree_with_max_4_children(edge_list):
    if not edge_list:
        return False  

    graph = defaultdict(list)
    nodes = set()

    for u, v in edge_list:
        graph[u].append(v)
        graph[v].append(u)
        nodes.add(u)
        nodes.add(v)

    if not is_connected(edge_list):
        return False

    if len(nodes) != 6:
        return False

    for node in graph:
        if len(graph[node]) > 4:
            return False
def generate_random_graph(n, cyclic=True):
    graph = Graph()

    for i in range(n):
        graph.add_node(i)

    connected_nodes = set()

    current_node = SC.randint(0, n - 1)
    connected_nodes.add(current_node)

    while len(connected_nodes) < n:
        possible_edges = []
        for node in connected_nodes:
            if graph.degree(node) < 4:  
                for neighbor in range(n):
                    if (neighbor not in connected_nodes and
                            not graph.edge_exists(node, neighbor) and
                            graph.degree(neighbor) < 4):
                        possible_edges.append((node, neighbor))

        if possible_edges:
            edge = SC.choice(possible_edges)
            graph.add_edge(edge[0], edge[1])
            connected_nodes.add(edge[1])
    if cyclic:

      potential_edges = list(itertools.combinations(range(n), 2))
      SC.shuffle(potential_edges)

      for edge in potential_edges:
          if (not graph.edge_exists(edge[0], edge[1]) and
                  graph.degree(edge[0]) < 4 and
                  graph.degree(edge[1]) < 4):
              graph.add_edge(edge[0], edge[1])

              break

    return graph

def edge_convert(num, lst):
    graph = Graph()
    for i in range(num):
        graph.add_node(i+1)
    for i in range(len(lst)):
        graph.add_edge(lst[i][0], lst[i][1])
    return graph

class TreeNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []

def tree_form(tabbed_strings):
    lines = tabbed_strings.split("\n")
    root = TreeNode("Root") 
    current_level_nodes = {0: root}
    stack = [root]
    for line in lines:
        level = line.count(' ') 
        node_name = line.strip() 
        node = TreeNode(node_name)
        while len(stack) > level + 1:
            stack.pop()
        parent_node = stack[-1]
        parent_node.children.append(node)
        current_level_nodes[level] = node
        stack.append(node)
    return root.children[0] 

def str_form(node):
    def recursive_str(node, depth=0):
        result = "{}{}".format(' ' * depth, node.name) 
        for child in node.children:
            result += "\n" + recursive_str(child, depth + 1) 
        return result
    return recursive_str(node)

def replace(equation, find, r):
    if str_form(equation) == str_form(find):
      return r
    col = TreeNode(equation.name, [])
    for child in equation.children:
      col.children.append(replace(child, find, r))
    return col

def remove_past(equation):
    coll = TreeNode(equation.name, [])
    for child in equation.children:
      if child.name == "del":
          for subchild in child.children:
              coll.children.append(remove_past(subchild))
      else:
          coll.children.append(remove_past(child))
    return coll

def break_equation(equation):
    sub_equation_list = [equation]
    equation = equation
    for child in equation.children: 
        sub_equation_list += break_equation(child) 
    return sub_equation_list

def multiple(equation):
    def split(equation):
        output = []
        for i in range(len(equation.children)-1):
            output.append(TreeNode("compound", [equation.children[i], equation.children[-1]]))
        return TreeNode("del", output)
    for item in break_equation(equation):
        if item.name == "compound" and len(item.children) > 2:
            equation = replace(equation, item, split(item))
    return remove_past(equation)

def pre(compound):
    global name_2

    compound = compound.replace("ane", "")

    for item in name:
        if item == compound:
            return "0*"+compound
        elif "cyclo"+item == compound:
            return "0*"+compound

    for i in range(10,-1,-1):
        compound = compound.replace(str(i)+"-", str(i)+"*")

    for item in name_2:
        compound = compound.replace(item, "")

    compound = compound.replace("yl", ";")

    compound = compound.replace(";)", ")")
    compound = compound.replace(";;", ";")
    compound = compound.replace(")cyclo", ");cyclo")
    compound = compound.replace(";cyclo", ";cyclo")

    for item in name:
        compound = compound.replace(";" + item, ";0*" + item)
    compound = compound.replace(";-", ";")
    compound = compound.replace(";cyclo", ";0*cyclo")
    compound = compound.replace(")-", ");")
    return compound
def name2compound(graph, name_eq, position):
    global name
    if name_eq.name != "segment":
        return name2compound(copy.deepcopy(graph), TreeNode("segment", [name_eq]), position)
    for child in name_eq.children:
        if int(child.children[0].name) == 0:
            graph = add_base(copy.deepcopy(graph), name.index(child.children[1].name.replace("cyclo", ""))+1, position, child.children[1].name.replace("cyclo", "") != child.children[1].name)

    for child in name_eq.children:
        if int(child.children[0].name) == 0:
            continue
        if child.children[1].name != "segment":
            graph = add_base(copy.deepcopy(graph), name.index(child.children[1].name.replace("cyclo", ""))+1, position+int(child.children[0].name), child.children[1].name.replace("cyclo", "") != child.children[1].name)
        else:
            graph2 = name2compound(Graph(), child.children[1], 0)
            max_node = sorted(graph.nodes, key=lambda x: -x[0])[0][0]
            for i in range(len(graph2.nodes)):
                graph2.nodes[i][0] += max_node
            for i in range(len(graph2.edges)):
                graph2.edges[i][0][0] += max_node
                graph2.edges[i][0][1] += max_node
            graph.nodes += graph2.nodes
            graph.edges += graph2.edges
            graph.add_edge(position+int(child.children[0].name), max_node+1)
    return graph

def add_base(graph, num, position, cyclic=False):
    max_node = 0
    if position != 0:
        max_node = sorted(graph.nodes, key=lambda x: -x[0])[0][0]

    for i in range(num):
        graph.add_node(max_node+i+1)
        if i != 0:
            graph.add_edge(max_node+i, max_node+i+1)
    if cyclic:
        graph.add_edge(max_node+1, max_node+num)
    if position != 0:
        graph.add_edge(position, max_node+1)
    return graph

def post(x):
    global name
    for item in name:
        x = x.replace("("+item+"yl)", item+"yl")
    return x
final = []

def string_equation_helper(equation_tree):
    if equation_tree.children == []:
        return equation_tree.name 
    s = "(" 
    if len(equation_tree.children) == 1:
        s = equation_tree.name + s

    sign = {"compound": "-", "segment": "?"}
    for child in equation_tree.children:
        s+= string_equation_helper(copy.deepcopy(child)) + sign[equation_tree.name]
    s = s[:-1] + ")"
    return s

def string_equation(eq): 
    eq = eq.replace("v_0", "x")
    eq = eq.replace("v_1", "y")
    eq = eq.replace("v_2", "z")
    eq = eq.replace("d_", "")

    return string_equation_helper(tree_form(eq))

st.title("Random Alkane Hydrocarbon Chemical Structure Generator")

num_carbons = st.slider("Select number of Carbon atoms:", 1, 40, 10)
is_cyclic = st.checkbox("Should the compound be cyclic?", value=False)

if st.button("Generate Structure"):

    graph = generate_random_graph(num_carbons, is_cyclic)

    iupac_name = process(graph)

    for item in name:
        iupac_name = iupac_name.replace(f"({item}yl)", f"{item}yl")

    st.write("IUPAC Name:", iupac_name)

    image = graph.draw_graph()

    st.image(image, caption="Chemical Structure", use_column_width=False)
