import pandas as pd
import argparse
import sys
import os
import fitness_functions as fit

# Enable/Disable debug
DEBUG = False


def print_traverse_path(path_list):
    print(f"Fitness function selected: {args.fitness}\n")
    print(f"Fitness summary: {selected_fit_function.__doc__}\n")
    print(f"ALPHA: {args.alpha}")
    if args.min_ones == args.max_ones:
        print(f"Increment: {args.min_ones} concession at each step")
    else:
        print(f"Increment: between {args.min_ones} and {args.max_ones} concessions at each step")

    print(f"[{len(path_list)}] Traverse Path{'s' if len(path_list) > 1 else ''} Identified")
    for path in path_list:
        print("----------- Path ----------- ")
        for step in range(0, len(path)):
            # the * in front refers to the panda unpacking of the value inside the DataFrame/Series
            if DEBUG:
                print("[{} - #{:<3}](P_uvalue256={:+2.9f}, I_uvalue256={:+2.9f})".
                      format(str(path[step]['Name']), str(path[step]['card']),
                             path[step]['P_uvalue256'], path[step]['I_uvalue256']))
            else:
                print("{0} {1}".format(str(path[step]['Name']), str(path[step]['card'])))

        print("\n")


def print_path_so_far(path):
    if len(path) == 0:
        print("\tNo cards in the current path")
    else:
        print("\tThe current path contains the following cards:")
        for step in range(0, len(path)):
            print("\t[{} - #{:<3}]".format(str(path[step]['Name']), str(path[step]['card'])))


def get_best_candidate_from_subset(curr_node, candidates_matrix, fitness_function):
    candidates_matrix["P_uRef"] = curr_node["P_uvalue256"]
    candidates_matrix["I_uRef"] = curr_node["I_uvalue256"]

    # Generates a tables with the utility for each row
    fitness_values = candidates_matrix.apply(fitness_function, axis=1)

    # obtain the max value of fitness, if there are no real values for fitness
    # select the min value of all the imaginary values of fitness.
    # The idea is that if we don't have improving fitness steps, we choose the least damaging for one party
    selected_fitness = fitness_values.max()

    if DEBUG:
        print("GET_BEST_CANDIDATE_FROM_SUBSET")

        print("Calculated fitness for each card (Utility values approx. to 9th decimal)\n"
              ">[{} - #{:<3}](P_uvalue256={:+2.9f}, I_uvalue256={:+2.9f}) <-- current card"
              .format(curr_node['Name'], curr_node['card'],
                      curr_node['P_uvalue256'], curr_node['I_uvalue256']))

        for idx, card in candidates_matrix.iterrows():
            print(".[{} - #{:<3}](P_uvalue256={:+2.9f}, I_uvalue256={:+2.9f}) > fitness:{}"
                  .format(card['Name'], card['card'],
                          card['P_uvalue256'], card['I_uvalue256'],
                          fitness_values.loc[idx]))

    # returns the rows such that the fitness value has the highest (least if the all the utilities are complex) utility
    return candidates_matrix.loc[fitness_values[fitness_values == selected_fitness].index]


# Returns a dataframe containing only the nodes that can be checked as next step in respect to current_node.
# As example: if current node has 3 1'sm this function will return all the nodes that have 4 1's of which 3 are in
# the same position as the 1's in the current_node
def obtain_candidate_nodes_for_next_iteration(curr_node, full_table_of_nodes,
                                              min_ones_for_candidates=1,
                                              max_ones_for_candidates=1):
    """

    :param curr_node: current node from which to identify candidates
    :param full_table_of_nodes: DataFrame with all the possible cards
    :param min_ones_for_candidates: [Optional. Default=1]
    :param max_ones_for_candidates: [Optional. Default=1]
    :return: Subset of the original DataFrame containing only the identified candidates
    """

    if DEBUG:
        print(f"\nOBTAIN_CANDIDATE_NODES_FOR_NEXT_ITERATION:"
              f"\ncurr_node [{curr_node['Name']} - #{curr_node['card']}]"
              f"\nAdditional 1s allowed: from [{min_ones_for_candidates}] to [{min_ones_for_candidates}]")

    curr_nodes_attr_only = curr_node[
        ["A_settlementfreeze", "B_jewishidentity", "C_landswaps", "D_6040waterrights", "E_accesstemple",
         "F_Jerusalemcapital", "G_Palestinianstate", "H_freedommovemt", "CountOfOnes"]]

    how_many_ones_in_current_node = int(curr_node.CountOfOnes)
    if DEBUG: print(f"The current card has {how_many_ones_in_current_node} 1s.\n"
                    f"Filtering cards with [{how_many_ones_in_current_node + min_ones_for_candidates}] "
                    f"to [{how_many_ones_in_current_node + min_ones_for_candidates}] 1s")

    #  The candidate_subset contains all the nodes with a number of 1s that is one unit bigger than the current
    # candidate_subset = full_table_of_nodes.loc[
    #     full_table_of_nodes['CountOfOnes'] == how_many_ones_in_current_node + 1] or included in the wanted range
    proposed_candidates_subset = full_table_of_nodes.loc[
        (full_table_of_nodes['CountOfOnes'] >= how_many_ones_in_current_node + min_ones_for_candidates) &
        (full_table_of_nodes['CountOfOnes'] <= how_many_ones_in_current_node + max_ones_for_candidates)
        ][["A_settlementfreeze", "B_jewishidentity", "C_landswaps", "D_6040waterrights", "E_accesstemple",
           "F_Jerusalemcapital", "G_Palestinianstate", "H_freedommovemt", "CountOfOnes"]]

    if DEBUG: print(f"> Candidates found: [{len(proposed_candidates_subset)}] cards") \
        if len(proposed_candidates_subset) > 0 else print(f"No Candidate cards found")

    if DEBUG and len(proposed_candidates_subset) > 0:
        print(f"> All Candidate cards:")
        for _, s in full_table_of_nodes.loc[proposed_candidates_subset.index].iterrows():
            print(f"{s['Name']} - {s['card']}")

    # We use multiplication because it will identify only those cards that have 1's in the same location with
    # the current reference card. The multiplication acts as a sieve that lets go through only those 1's where
    # there is a 1 in the current_card.
    # Given the assumption that the candidates subset contains only nodes having one 1
    # more than the current reference card, we can select all those cards that have the result of this multiplication
    # giving a number of 1's equal to the reference card. Ex:
    #  ref_card = 0100010
    #  candidate_subset =
    #  0110010
    #  0111000
    #  1100010
    #  0101001
    # The result of the multiplication by element will return :
    #  result_of_mul:
    #  0100010 <---
    #  0100000
    #  0100010 <---
    #  0100000
    candidates_over_reference = proposed_candidates_subset.mul(curr_nodes_attr_only.values)

    # we will retrieve the index of the cards that satisfy our selection criteria (same numbers of 1's after mul
    # as ref_card) and we will use these ids to create a view of the original subset with only the selected cards
    candidate_subset = \
        candidates_over_reference.loc[
            (candidates_over_reference['A_settlementfreeze'] +
             candidates_over_reference['B_jewishidentity'] +
             candidates_over_reference['C_landswaps'] +
             candidates_over_reference['D_6040waterrights'] +
             candidates_over_reference['E_accesstemple'] +
             candidates_over_reference['F_Jerusalemcapital'] +
             candidates_over_reference['G_Palestinianstate'] +
             candidates_over_reference['H_freedommovemt']) == how_many_ones_in_current_node]

    if DEBUG and len(proposed_candidates_subset) > 0:
        print(f"> Selected Candidate cards from [{curr_node['Name']} - #{curr_node['card']}]:")
        for _, s in full_table_of_nodes.loc[candidate_subset.index].iterrows():
            print(f"{s['Name']} - {s['card']}")

    return full_table_of_nodes.loc[candidate_subset.index]


def walk_through(curr_node,
                 ref_node,
                 full_table_of_nodes,
                 current_traverse_path,
                 all_paths,
                 current_step_number,
                 fitness_fnc,
                 min_ones_for_candidates=1,
                 max_ones_for_candidates=1):
    """
    This function walks along the graph formed by all the possible cards starting from curr_node used as initial node.
    curr_node can be any node in the DataFrame. The fitness function used to identify the cards to chose from
    the current one is passed as inpute parameter fitness_fnc.
    The class fitness_function.FitnessFunctions contains a set of available fitness functions that can be used.

    It is possible to configure how many ones in addition to the ones already present in the current card are allowed
    through the parameters `min_ones_for_candidates` that defined the minimum number of ones to look for, and
    `max_ones_for_candidates` that ddefines the maximum number of additional ones are possible in the set of
    candidate cards

    :param curr_node: card from which start the traversing
    :param ref_node: not used currently; could be used to pass a card to use as reference in fitness calculation
    :param full_table_of_nodes: Dataframe containing all the available cards. Loaded from csv
    :param current_traverse_path: list object used to contain the currently worked path
    :param all_paths: list object that will store all the identified paths
    :param current_step_number: Used to keep track of the depth of the traverse path. Mainly used for printing
    :param fitness_fnc: Fitness function that will be used
    :param min_ones_for_candidates: Min number of additional one to look for in candidate steps. Default = 1
    :param max_ones_for_candidates: Max number of additional one to look for in candidate steps. Default = 1

    :return: The path identified
    """

    if DEBUG:
        print(f"\nWALK_TROUGH {current_step_number}: Current Node: {curr_node['Name']} - Card #{curr_node['card']}")
        print_path_so_far(current_traverse_path)

    # Storing the actual node to the current path
    current_traverse_path.append(curr_node)

    if DEBUG: print(f"\tFiltering candidate cards for next iteration...")
    next_step_candidates = obtain_candidate_nodes_for_next_iteration(curr_node, full_table_of_nodes,
                                                                     min_ones_for_candidates,
                                                                     max_ones_for_candidates)

    if len(next_step_candidates) > 0:
        if DEBUG: print(f"Comparing candidate Cards...")
        best_next_steps = get_best_candidate_from_subset(curr_node, next_step_candidates, fitness_fnc)

        if DEBUG and len(best_next_steps) > 0:
            print(f"\n> Chosen cards from [{curr_node['Name']} - #{curr_node['card']}]:")
            for _, s in full_table_of_nodes.loc[best_next_steps.index].iterrows():
                print(f"{s['Name']} - {s['card']}")

        # It can be theoretically possible to have multiple viable options from the current position
        # so we are going to explore all possible paths
        row_counter = 0
        # current_path_snapshot will be used as a copy of the current path to copy when multiple paths are found
        current_path_snapshot = list(current_traverse_path)
        path = current_traverse_path

        for _, next_step in best_next_steps.iterrows():
            # if there are more than one possible path, a copy of the current path is created
            # and a new possible path to traverse added
            if row_counter > 0:
                print(f"A new path was identified after step {current_step_number + 1}. Starting its traversal")
                # cloning the current path
                path = list(current_path_snapshot)
                # adding the new cloned path to the list of known paths
                all_paths.append(path)

            row_counter += 1
            walk_through(curr_node=next_step, ref_node=ref_node, full_table_of_nodes=full_table_of_nodes,
                         current_traverse_path=path, all_paths=all_paths,
                         current_step_number=current_step_number + 1,
                         fitness_fnc=fitness_fnc,
                         min_ones_for_candidates=min_ones_for_candidates,
                         max_ones_for_candidates=max_ones_for_candidates)

    return current_traverse_path


def load_csv(csv_path):
    # Loading initial CSV file
    # csv_filename = "/Users/xxxx/Documents/cardgame_graph_traversal/cards-attributes-utilities-256cards.csv"
    # csv_filename = "test_data/cards-attrib-util-256_there_is_one_path.csv"

    if not os.path.isfile(csv_path):
        print("Couldn't find the provided filename: ", csv_path)
        sys.exit("The provided csv filename was not found on the system.")

    print("\nLoading csv: ", csv_path)
    df_from_csv = pd.read_csv(csv_path)
    # Print how many rows, columns
    rows, cols = df_from_csv.shape
    print(f"\tLoad complete. Loaded {rows} rows and {cols} columns")

    return df_from_csv


def count_and_name_cards(data_frame):
    # Using the binary representation of each card as its name
    if DEBUG == True:
        print("\tGenerating the name of each node based on the 0,1 configuration")
    data_frame['Name'] = "card_" + \
                         data_frame['A_settlementfreeze'].map(str) + \
                         data_frame['B_jewishidentity'].map(str) + \
                         data_frame['C_landswaps'].map(str) + \
                         data_frame['D_6040waterrights'].map(str) + \
                         data_frame['E_accesstemple'].map(str) + \
                         data_frame['F_Jerusalemcapital'].map(str) + \
                         data_frame['G_Palestinianstate'].map(str) + \
                         data_frame['H_freedommovemt'].map(str)

    if DEBUG: print("\tGenerating the column with the count of 1's for each card/row")
    data_frame['CountOfOnes'] = data_frame['A_settlementfreeze'] + data_frame['B_jewishidentity'] + \
                                data_frame['C_landswaps'] + data_frame['D_6040waterrights'] + \
                                data_frame['E_accesstemple'] + data_frame['F_Jerusalemcapital'] + \
                                data_frame['G_Palestinianstate'] + data_frame['H_freedommovemt']


# Collect the available fitness functions to show in the command line help
available_fitness_fn = []
for fn_name in dir(fit.FitnessFunctions()):
    if not fn_name.startswith("__") and fn_name != 'ALPHA':
        available_fitness_fn.append(fn_name)

# Instantiate the parser
parser = argparse.ArgumentParser(
    description='Traverse the given set of cards using a set of fitness functions '
                'that can be specified as start parameters')
parser.add_argument('--csv_file_name', required=True,
                    help="Path of the csv file to load")
parser.add_argument('--fitness', choices=available_fitness_fn, required=True,
                    help="Fitness function to use to identify the next viable card")
parser.add_argument('--start_card', default='card_00000000',
                    help="Initial card to start the path from. Default: %(default)s")
parser.add_argument('--min_ones', type=int, default=1,
                    help="Min number of additional one to look for in candidate cards. Default: 1")
parser.add_argument('--max_ones', type=int, default=1,
                    help="Max number of additional one to look for in candidate cards. Default: 1")
parser.add_argument('--alpha', type=float, default=.5,
                    help="ALPHA coefficient for the bargaining power")
parser.add_argument('--debug', action='store_true',
                    help="Enable Debug output")

args = parser.parse_args()

if args.debug:
    DEBUG = True
    print("\nDEBUG ON")

# df = load_csv("/Users/gtrovato/Documents/Playground/cardgame_graph_traversal/cards-attributes-utilities-256cards.csv")
df = load_csv(args.csv_file_name)
count_and_name_cards(df)

# In the traversing we select the current node and identify the subset of nodes that can
# be reached from the current node. Among these subset of nodes, the node with the highest utility
# will be selected as next node and it will become the next current node
# for which calculate the following node

# Selecting the status quo ==> 00000000 as starting node
# start_node_name = 'card_00000000'
start_node_name = args.start_card

start_node = df.loc[df['Name'] == start_node_name]
print(f"\nStarting traversing from [{start_node_name}]")

# Selecting the reference node as the starting point. This node will stay the same in this version of the traverse.
#  This was a design choice. If we want to calculate the utility of the node to select in  relation to the
# reference node, this node can be updated accordingly.
reference_node = start_node

#  The traverse path is a list that contains the name of the nodes identified to satisfy the path
# It is initialised with the initial start_node
final_traverse_paths = []
initial_traverse_path = []
final_traverse_paths.append(initial_traverse_path)

# Initialized the Fitness functions package
fit_fnc = fit.FitnessFunctions()
# It is possible to change the value of ALPHA used by the Fitness functions used from this object.
# If not set, default value is 0.5
fit_fnc.ALPHA = args.alpha
# obtains the function handler to use as fitness function
selected_fit_function = getattr(fit_fnc, args.fitness)

if DEBUG: print(f"fitness function: [{args.fitness}]\n Summary: {selected_fit_function.__doc__}")

# Using start_node.squeeze() so that at each iteration the walk_through function
# is receiving a pandas.Series object instead than a pandas.DataFrame

# This call starts the walk through the data points of the card game
walk_through(curr_node=start_node.squeeze(),  # card from which start the traverse
             ref_node=reference_node,
             # not used currently; could be used to pass a card to use as reference in fitness
             full_table_of_nodes=df,  # Dataframe containing all the available cards. Loaded from csv
             current_traverse_path=initial_traverse_path,  # list object used to contain the currently worked path
             all_paths=final_traverse_paths,  # list object that will store all the identified paths
             current_step_number=0,  # Used to keep track of the depth of the traverse path. Mainly for printing
             fitness_fnc=selected_fit_function,  # fitness function that will be used
             min_ones_for_candidates=args.min_ones,  # Min number of additional one to look for in candidate steps
             max_ones_for_candidates=args.max_ones)  # Max number of additional one to look for in candidate steps

print_traverse_path(final_traverse_paths)
