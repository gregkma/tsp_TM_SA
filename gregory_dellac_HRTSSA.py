import numpy as np
from geopy.distance import great_circle
import pandas as pd
import copy
import folium
from ast import literal_eval
from numpy.random import randint


"""
 Guidelines reminders:
 -One time per town
 -Smallest distance possible
 -Going through all towns in the given list
 -Start at Shanghai
 -Distance in km

 Glossary:
 -temp = temperature
 -s_a = simulated annealing
 -t_s = tabu search
 -inf = inferior
 -sup = superior
"""
# hyper-parameters

# length of path and number of swap inF
nb_town = 100  # select an int, number of town where the salesman goes

# simulated_annealing
alpha = 0.99  # must be inferior to 1, decrement of temp through s_a algorithm
temp_i = 50 # any value sup to 0, initial temperature of s_a algorithm
temp_f = 0.15 # any value sup to 0, final temperature of s_a algorithm

# variables

# lists and arrays

# save best result
best_dist = np.inf  # initialise best distance set to +infinity
best_path = np.zeros(nb_town - 1)
old_best_path = np.zeros(nb_town - 1)

# tabu index for forbidden move
tabu_list = []
tabu_index = []

# upload data
data = pd.read_csv(
    r'D:\Documents\Etude\inge\2A\IA\Python/geonames-all-cities-with-a-population-1000.csv',
    delimiter=';')

# select the 100 first most populated towns
data = data.sort_values(by="Population", ascending=False)
data = data.iloc[:nb_town, :]

# selecting useful information
coordinates = data["Coordinates"]
coordinates.reset_index(drop=True, inplace=True)
name_town = data['Name']
name_town.reset_index(drop=True, inplace=True)

# create dataframes and reset the index
town = pd.concat([name_town, coordinates], axis=1)
town.reset_index(drop=True, inplace=True)

# remove Shanghai from the list
town_without_shanghai = town.iloc[1:]
town_without_shanghai.reset_index(drop=True, inplace=True)
town['Coordinates'] = town['Coordinates'].apply(literal_eval)

#initialise
shanghai = town.iloc[0]

#functions

def random_generation_path(matrix):
    # generates a route randomly
    random_town = np.random.choice(matrix.index, replace=False, size=len(matrix))
    return random_town


def generate_path_distance(initialise, matrix, random_town):
    # initialization
    arc_l = []
    dist = great_circle(initialise['Coordinates'], matrix.iloc[random_town[0]]['Coordinates']).km
    arc_l.append(dist)
    # total distance calculation and in nodes calculation
    for i in range(1, len(random_town)):
        arc = great_circle(matrix.iloc[random_town[i - 1]]['Coordinates'], matrix.iloc[random_town[i]]['Coordinates']).km
        arc_l.append(arc)
        dist = dist + arc
    return random_town, int(dist), arc_l


def sort_solution(candidates, initialise, matrix):
    # function to get the distance for a given route

    def route_distance(route):
        _, dist, _ = generate_path_distance(initialise, matrix, route)
        return dist
    
    # finding the route with the minimum distance
    min_route = min(candidates, key=route_distance)
    min_dist, min_arc = generate_path_distance(initialise, matrix, min_route)[1:]
    
    return min_route, min_dist, min_arc

    
def get_gamma(l_arc, dist):
    # calculating degree of irregularity
    gamma = (nb_town * np.std(l_arc))/ (dist)
    return gamma

def main(initialise=shanghai, matrix=town_without_shanghai):
    global alpha, temp_i # hyper-parameters
    global tabu_list, tabu_index  # algorithms variables
    global best_path, best_dist, old_best_path  # best result variables

    # initialisation by generating a random path
    random_town = random_generation_path(matrix)
    route, dist, arc_l = generate_path_distance(initialise, matrix, random_town)
    route = route.tolist()

    gamma = get_gamma(arc_l, dist)
    

    def get_hp():
        elen = int((5600*(nb_town**0.4)*(1.27+(gamma**4.11))*(4.72*(10**-11)*(gamma+0.1)+(nb_town**-2.81)))/((22.1+(gamma**4.11))*(1.42*(10**-11)+(nb_town**-2.81))))  # epoch length
        tabu_tenure = int(((elen**0.6)/3.5))  # max number of tabu moves
        C_N = int(((2800*(nb_town**1.1))/elen))  # number of solutions proposed per temperature
        return elen, tabu_tenure, C_N

    elen, tabu_tenure, C_N = get_hp()  # getting the rest of hyper-parameters
    
    """
    creating the functions we need to apply the combinated method of research
    """
    

    def swap_2_opt(route):
        # applying 2-opt algorithm 
        list_candidates = []
        # generating solutions for a temperature
        for i in range(C_N):
            # creating nodes for segmentation
            node1 = 0
            node2 = 0
            # giving values to the nodes
            while node1 == node2:
                node1 = randint(1, len(route)-1)
                node2 = randint(1, len(route)-1)
                # swap procedure
                if node1 > node2:
                    swap = node1
                    node1 = node2
                    node2 = swap
                # reorganizing the route
                tmp = route[node1:node2]
                tmp_bis = route[:node1] + tmp[::-1] + route[node2:]
                # saving results
                list_candidates.append(tmp_bis)
        return list_candidates
    

    def tm_verify_length():
        # verify if the length of tabu_list is respected
        if len(tabu_list) > tabu_tenure:
            return False


    def tm_test(dist, route):
        global tabu_list, tabu_index
        # testing tabu moves
        for i in range(len(tabu_list)):
            if route[tabu_index[i]] == tabu_list[i]:
                return False
        return True


    def tm_update(old_best, actual_best):
        global tabu_list, tabu_index
        # updating the tabu moves list
        for i in range(len(actual_best)):
            if old_best[i] != actual_best[i]:
                tabu_list.append(old_best[i])
                tabu_index.append(i)
        # reset tabu list if stopping criteria is reached
        if len(tabu_list) > tabu_tenure:
            while len(tabu_list) - tabu_tenure != 0:
                if tabu_list != []:
                    tabu_list.pop(0)
                    tabu_index.pop(0)
        return tabu_list, tabu_index


    def get_probability(new, current):
        global temp_i
        # getting probability for the s_a algorithm logic
        if new - current <= 0:
            probability = -2.46 * nb_town * (new - current) / (temp_i * new * (3.7 + (gamma ** 1.1)))
        else:
            probability = 1
        return probability
    

    def sa(new_dist, current_dist, alpha):
        global temp_i
        # s_a algorithm
        delta_distance = new_dist - current_dist
        probability = get_probability(new_dist, current_dist)
        if probability < np.exp((-delta_distance) / temp_i):
            return True
        else:
            return False

    # creation of the current solution
    current_dist = copy.deepcopy(dist)
    current_path = copy.deepcopy(route)

    # loop
    while temp_i > temp_f:
        # to avoid being stuck in a temperature when result can't be improved

        num_iter = int(elen / 10)
        for i in range(num_iter):
            print('--------')
            print('temperature actuelle:',temp_i)
            print('iteration numero:', i, 'sur', num_iter)
            # applying 2-opt-swap algo and selecting the best result and calculating distance
            candidates = swap_2_opt(route)
            route, dist, arc_l = sort_solution(candidates, initialise, matrix)
            
            # recalculating hyper-parameters    
            gamma = get_gamma(arc_l, dist)
            elen, tabu_tenure, C_N = get_hp()  # hyper-parameters

            # aspiration criteria
            if dist < best_dist:
                # updating results
                old_best_path = copy.deepcopy(best_path)
                best_dist = copy.deepcopy(dist)
                best_path = copy.deepcopy(route)
                # updating tabu_list as results are being improved
                tabu_list, tabu_index = tm_update(old_best_path, best_path)

            # t_s algorithm
            elif not tm_verify_length():
                if tm_test(dist, route):
                    # s_a algorithm
                    if sa(dist, current_dist, alpha):
                        # updating results
                        current_dist, current_path = dist, route
                        tabu_list, tabu_index = tm_update(current_path, route)

            # checking if results are better, aspiration criterion
            if current_dist < best_dist:
                # updating results
                old_best_path = copy.deepcopy(best_path)
                best_dist = copy.deepcopy(current_dist)
                best_path = copy.deepcopy(current_path)
                # updating tabu_list as results are being improved
                tabu_list, tabu_index = tm_update(old_best_path, best_path)

            # printing result while looping
            print('--------')
            print('meilleure distance actuelle:', best_dist)
            print('nombre de tabou:', len(tabu_list))

        temp_i *= alpha

    return best_dist, best_path


def tsp_sol(initialise=shanghai, matrix=town_without_shanghai):
    global alpha, temp_i  # hyper-parameters
    global deny, accept, tabu_list, tabu_index  # algorithms variables
    global best_path, best_dist, old_best_path  # best result variables

    # calling in results
    # best_dist, best_path = main(alpha, beta, temp, nb_swap, initialise, matrix)

    best_dist = 103020
    best_path = np.array([5, 34, 76, 78, 77, 1, 16, 2, 21, 33, 79, 62, 81, 29, 82, 85, 20, 41, 91, 66,
                          12, 53, 19, 88, 18, 70, 69, 73, 0, 68, 95, 36, 55, 90, 10, 50, 57, 38, 93, 15,
                          83, 27, 80, 51, 54, 25, 45, 22, 28, 52, 64, 74, 14, 61, 60, 26, 39, 96, 6, 63,
                          42, 11, 44, 65, 9, 87, 71, 35, 37, 13, 48, 89, 4, 86, 98, 49, 17, 67, 59, 47,
                          46, 97, 58, 40, 7, 56, 31, 32, 8, 75, 24, 92, 94, 43, 3, 84, 72, 30, 23])

    # correcting index
    best_path = [i + 1 for i in best_path]

    # adding Shanghai
    best_path = np.insert(best_path, 0,0).tolist()

    print(best_path)
    print(best_dist)

    results = folium.Map(location=[0, 0], zoom_start=2)

    title_text = f"Chemin le plus court pour {nb_town} villes d'une distance totale de {best_dist} km"

    title_html = f'''
    <h1 style="font-size:24px; font-weight:bold; text-align:center;">
    {title_text}
    </h1>
    '''

    towns = []
    for i in best_path:
        coord = town.iloc[i]['Coordinates']
        towns.append(coord)

    results.get_root().html.add_child(folium.Element(title_html))

    folium.Marker(shanghai['Coordinates'], tooltip="Shanghai").add_to(results)

    for coord in towns:
        folium.CircleMarker(location=coord, radius=5, color='blue').add_to(results)

    folium.PolyLine(locations=[towns[i] for i in best_path], color='green', weight=2.5, opacity=1).add_to(results)

    results.save("output_map.html")
    return towns