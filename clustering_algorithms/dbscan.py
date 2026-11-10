#lets code again 5:27pm already 3hr+ spend but I don't lose

#labels 0,1,2..-> cluster and -1->noise
labels = np.zeros(len(X))
eps = 0.2
min_samples = 5
cluster = set()
point_idx = np.random.randint(len(X))
point = X[point_idx]

def find_neighbors(X,point_idx,eps):
    distances = np.linalg.norm(X - X[point_idx],axis=1)
    return np.where(distances <= eps)[0]

neighbors = find_neighbors(X,point_idx,eps)
print(len(neighbors)) #contain index of X

cluster.update(neighbors)
if len(neighbors) >= min_samples:
    seeds = list(neighbors)
    visited = set() #AI help
    while(seeds):
        neighbor = seeds.pop()
        if neighbor in visited:
            continue
        visited.add(neighbor)
        #print(neighbor)
        new_neighbors = find_neighbors(X,neighbor,eps)
        if len(new_neighbors) >= min_samples:
            #print("second")
            for idx in new_neighbors:
                cluster.add(idx) 
                if idx not in seeds and idx not in visited:
                    seeds.append(idx)
   
else:
    
    print("noise")