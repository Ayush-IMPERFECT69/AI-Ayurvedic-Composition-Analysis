import pandas as pd
import networkx as nx

def build_hsa_graph():
    print("Loading datasets...")
    # Load the CSV files 
    df_herb_ing = pd.read_csv('herb_ingredient.xlsx - Sheet1.csv')
    df_ing_tgt = pd.read_csv('ingredient_target.xlsx - Sheet1.csv')
    df_sym_tgt = pd.read_csv('symptom_target.xlsx - Sheet1.csv')
    df_herb_sym = pd.read_csv('herb_symptom.xlsx - Sheet1.csv')

    # Initialize a generic graph
    G = nx.Graph()

    print("Building the graph structure...")
    
    # 1. Add Herb-Ingredient Edges
    for _, row in df_herb_ing.iterrows():
        herb, ingredient = row['Herb'], row['Ingredient']
        G.add_node(herb, type='Herb')
        G.add_node(ingredient, type='Ingredient')
        G.add_edge(herb, ingredient, edge_type='contains')

    # 2. Add Ingredient-Target Binary Interactions
    for _, row in df_ing_tgt.iterrows():
        ingredient, target = row['Ingredient'], row['Target']
        G.add_node(ingredient, type='Ingredient')
        G.add_node(target, type='Target')
        G.add_edge(ingredient, target, edge_type='interacts_with')

    # 3. Add Symptom-Target Associations
    for _, row in df_sym_tgt.iterrows():
        symptom, target = row['Symptom'], row['Target']
        G.add_node(symptom, type='Symptom')
        G.add_node(target, type='Target')
        G.add_edge(symptom, target, edge_type='associated_with')

    # 4. Add Herb-Symptom Ground Truth Labels (for supervised learning)
    # We will store the association label (0 or 1) as an edge attribute
    for _, row in df_herb_sym.iterrows():
        herb, symptom, label = row['Herb'], row['Symptom'], row['Label']
        G.add_node(herb, type='Herb')
        G.add_node(symptom, type='Symptom')
        
        # Only add the edge if there is a positive association, or store it as an attribute
        if label == 1:
            G.add_edge(herb, symptom, edge_type='treats', label=1)

    print("Graph construction complete!")
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Total Edges: {G.number_of_edges()}")
    
    return G, df_herb_sym

# Execute the pipeline
if __name__ == "__main__":
    hsa_graph, ground_truth_df = build_hsa_graph()