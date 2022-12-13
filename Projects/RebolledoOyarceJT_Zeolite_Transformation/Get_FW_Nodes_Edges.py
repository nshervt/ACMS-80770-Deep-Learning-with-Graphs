import numpy as np



def Get_Neighbors(Node, atoms):
    
    NumberElements = np.array(atoms.get_chemical_symbols())
    # NumberElements = np.array(atoms.get_chemical_symbols())

    # Identify the index of Oxygen and Silicon
    Position_Element_Si = np.where(NumberElements == 'Si')[0].copy()
    Position_Element_O  = np.where(NumberElements == 'O')[0].copy()
    Position_Element_Al  = np.where(NumberElements == 'Al')[0].copy()
    
    Position_Element_Si_Al = Position_Element_Si.copy()
    if len(Position_Element_Al) > 0: 
        for i in Position_Element_Al:
            Position_Element_Si_Al = np.append(Position_Element_Si_Al, i)
    
    Distances_sort = []
    Neighbor_index = []
    
    if Node in Position_Element_Si_Al:
        for i in Position_Element_O:
            current_distance = atoms.get_distance(Node,i, mic = True).copy()
            
            Distances_sort.append(current_distance)
            
        Distances = np.array(Distances_sort.copy())
        Distances_sort.sort()
        
        Neighbor1 = np.where(Distances == Distances_sort[0])[0][0]
        Neighbor2 = np.where(Distances == Distances_sort[1])[0][0]
        Neighbor3 = np.where(Distances == Distances_sort[2])[0][0]
        Neighbor4 = np.where(Distances == Distances_sort[3])[0][0]
        
        Neighbor1 = Position_Element_O[Neighbor1]
        Neighbor2 = Position_Element_O[Neighbor2]
        Neighbor3 = Position_Element_O[Neighbor3]
        Neighbor4 = Position_Element_O[Neighbor4]
        
        Neighbor_index.append(Neighbor1)
        Neighbor_index.append(Neighbor2)
        Neighbor_index.append(Neighbor3)
        Neighbor_index.append(Neighbor4)
            
    if Node in Position_Element_O:
        for i in Position_Element_Si_Al:
            current_distance = atoms.get_distance(Node,i, mic = True).copy()
            
            Distances_sort.append(current_distance)
        
        Distances = np.array(Distances_sort.copy())
        Distances_sort.sort()
        
        Neighbor1 = np.where(Distances == Distances_sort[0])[0][0]
        Neighbor2 = np.where(Distances == Distances_sort[1])[0][0]
        
        Neighbor1 = Position_Element_Si_Al[Neighbor1]
        Neighbor2 = Position_Element_Si_Al[Neighbor2]
        
        Neighbor_index.append(Neighbor1)
        Neighbor_index.append(Neighbor2)
    

    return Neighbor_index


def Get_Neighbors_Os_and_Ts(Node, atoms, tsites_flag = False):

    """
    Inputs:
    
    Node: Index to verify its neighbors

    Atoms: ASE Objects with the position of atoms

    Optional inputs:

    tsites_flag = Boolean, True = Return Os and Ts neighbors even when the node is not a t-site
                  False = Return an error if you input not a t-site index
                  
    Outputs:
    
    Neighbor_Os = list of index of Os neighbor
    Neighbor_Ts = list of index of Ts neighbor

    
    """

    NumberElements = np.array(atoms.get_chemical_symbols())
    # NumberElements = np.array(atoms.get_chemical_symbols())

    # Identify the index of Oxygen and Silicon
    Position_Element_Si = np.where(NumberElements == 'Si')[0].copy()
    Position_Element_O  = np.where(NumberElements == 'O')[0].copy()
    Position_Element_Al  = np.where(NumberElements == 'Al')[0].copy()
    
    
    
    Position_Element_Si_Al = Position_Element_Si.copy()
    if len(Position_Element_Al) > 0: 
        for i in Position_Element_Al:
            Position_Element_Si_Al = np.append(Position_Element_Si_Al, i)
        
        
        
    if Node in Position_Element_Si_Al:
        Neighbor_Os = Get_Neighbors(Node, atoms)
        
        Neighbor_Ts = []
        
        for i in Neighbor_Os:
            Neighbor_Os_aux = Get_Neighbors(i, atoms)
            if Neighbor_Os_aux[0] == Node:
                Neighbor_Ts.append(Neighbor_Os_aux[1])
            if Neighbor_Os_aux[1] == Node:
                Neighbor_Ts.append(Neighbor_Os_aux[0])
                
                
    elif Node in Position_Element_O and tsites_flag == True:
        
        Neighbor_Ts = Get_Neighbors(Node, atoms)
        
        Neighbor_Os = []
        
        for i in Neighbor_Ts:
            Neighbor_Ts_aux = Get_Neighbors(i, atoms)
            
            for j in Neighbor_Ts_aux:
                Neighbor_Os.append(j)
                
    else:
        raise ValueError("The index must be a T-site")

        


    return Neighbor_Os, Neighbor_Ts



def Get_Edges(atoms):

    NumberElements = np.array(atoms.get_chemical_symbols())
    # NumberElements = np.array(atoms.get_chemical_symbols())

    # Identify the index of Oxygen and Silicon
    Position_Element_Si = np.where(NumberElements == 'Si')[0].copy()
    Position_Element_O  = np.where(NumberElements == 'O')[0].copy()
    Position_Element_Al  = np.where(NumberElements == 'Al')[0].copy()
    
    
    
    Position_Element_Si_Al = Position_Element_Si.copy()
    if len(Position_Element_Al) > 0: 
        for i in Position_Element_Al:
            Position_Element_Si_Al = np.append(Position_Element_Si_Al, i)


    ## Extract the connection between Silicons and Aluminums (we only care about Si and Al because the active sites is located in those atoms)
    Edges_total = []
    for i in Position_Element_Si_Al:
        Neighbor_Os, Neighbor_Ts = Get_Neighbors_Os_and_Ts(i, atoms)
        
        for j in Neighbor_Ts:
            Edges = [i, j]
            Edges.sort()
            
            if Edges not in Edges_total and i != j:
                Edges_total.append(Edges)
    
    return Edges_total
