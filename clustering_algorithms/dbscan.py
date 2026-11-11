class DBscan:
    def __init__(self,min_points=10,eps = 0.2):
        self.min_points = min_points
        self.eps = eps

    def find_neighbors(self,X,point_idx,eps):
        distances = np.linalg.norm(X - X[point_idx],axis=1)
        return np.where(distances <= eps)[0]

    def fit(self,X):
        labels = np.full(len(X),-2)
        visited = set()
        cluster_id = 0

        for point_idx in range(len(X)):
            if point_idx in visited:
                continue
            visited.add(point_idx)
            neighbor = self.find_neighbors(X,point_idx,self.eps)
            #print(neighbor)
            #noise or core
            # print(len(neighbor))
            if len(neighbor) < self.min_points:
                #noise
                # print("noise")
                labels[point_idx] = -1
                continue

            if len(neighbor) >= self.min_points:
                #core point then so according to cluster value
                labels[point_idx] = cluster_id
                seeds = list(neighbor)

                while seeds:
                    neighbor = seeds.pop()
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_neighbor = self.find_neighbors(X, neighbor, self.eps)
                        #if new_neighbor core or noise
                        if len(new_neighbor) < self.min_points:
                            #noise
                            labels[neighbor] = -1
                        if len(new_neighbor) >= self.min_points:
                            #core point
                            #labels[point_idx] = cluster_id
                            #what if a new neighbor is let to investigate
                            for idx in new_neighbor:
                                if idx not in seeds and idx not in visited:
                                    seeds.append(idx)
                                    #print("append")
                    if labels[neighbor] in (-2, -1): #AI help
                        labels[neighbor] = cluster_id
            cluster_id +=1
        print(labels)
        return labels