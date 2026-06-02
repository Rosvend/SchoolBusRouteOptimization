import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

place_name = "Medellin, Colombia"
file_path = "medellin_drive.graphml"

def generador():
    if os.path.exists(file_path):
        print("📂 Cargando grafo desde archivo local...")
        G = ox.load_graphml(file_path)
    else:
        print("🌍 Descargando grafo desde OpenStreetMap...")
        G = ox.graph_from_place(place_name, network_type="drive", simplify=True)
        ox.save_graphml(G, file_path)
        print("💾 Grafo guardado localmente.")

    north = 6.312
    south = 6.182
    east  = -75.531
    west  = -75.626
    nodos=[n for n in G.nodes if G.nodes[n]['x']>=west and G.nodes[n]['x']<=east and G.nodes[n]['y']>=south and G.nodes[n]['y']<=north]
    G2=G.subgraph(nodos)
    #Parámetros del problema
    cap=20 #Número de niños por bus
    N=400 #Número de niños

    r_origen=(-75.58683269546758, 6.243283526198216)
    x=[G2.nodes[n]['x'] for n in G2.nodes]
    y=[G2.nodes[n]['y'] for n in G2.nodes]
    n=list(G2.nodes)

    dist=[(xj-r_origen[0])**2+(yj-r_origen[1])**2 for xj,yj in zip(x,y)]
    origen=n[np.argmin(dist)]
    #Voy a agregar los nodos que tengan por lo menos ruta desde y hacia el nodo origen
    print('Agregando los nodos aleatorios')
    ni=[]
    d_org={}
    while len(ni)<N:
        nodo=np.random.choice(nodos)
        if nx.has_path(G2, origen, nodo) and nx.has_path(G2, nodo, origen):
            ni.append(nodo)
            d_org[int(nodo)]=nx.shortest_path_length(G2, origen, nodo, weight="length")
            print(len(ni),',', end='')
            if(len(ni)%40==0):
                print('')
        ni.append(origen) #Agrego el nodo origen a la lista de nodos donde se ubican los niños, para asegurar que el nodo origen esté incluido en el escenario.
    xi=[G2.nodes[n]['x'] for n in ni]
    yi=[G2.nodes[n]['y'] for n in ni]

    latMin=south
    latMax=north
    lonMin=west
    lonMax=east

    # Distancias calculadas por Dijkstra
    dist={}
    for i in tqdm(range(len(ni))):
        distancias = nx.single_source_dijkstra_path_length(G2,source=ni[i], weight="length")
        for j in range(len(ni)):
            if(ni[j] in distancias.keys()):
                dist[(int(ni[i]), int(ni[j]))]=float(distancias[ni[j]])
            else:
                print(f"No hay camino entre {ni[i]} y {ni[j]}")
                dist[(int(ni[i]), int(ni[j]))]=d_org[ni[i]]+d_org[ni[j]] 

    return  G2, ni, xi, yi, dist

G, ni, xi, yi, dist = generador()