import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import QuantileTransformer

from . import VisualizationPipeline


class SOMVisualizationPipeline(VisualizationPipeline):
    def __init__(self):
        super().__init__()
        self._parameters = None
        self._ordering = None        


    def fit(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        """
        Note that som assumes datasets features to be objects that needs to be placed at a grid. (i.e., columns, not rows)
        Upon feature positioning, individual objects are visualized in a fairly standard way

        Parameters
        ----------
        n : int, default=3
            The SOM's shape (we only consider square SOMs for simplicity).
        dim : int, default=3
            The dimensionality (number of features) of the input space.
        lr : float, default=1
            The initial step size for updating the SOM weights.
        sigma : float, optional
            Optional parameter for magnitude of change to each weight. Does not
            update over training (as does learning rate). Higher values mean
            more aggressive updates to weights.
        max_iter : int, optional
            Optional parameter to stop training if you reach this many
            interation.
        random_state : int, optional
            Optional integer seed to the random number generator for weight
            initialization. This will be used to create a new instance of Numpy's
            default random number generator (it will not call np.random.seed()).
            Specify an integer for deterministic results.
        som_type : string, optional
            Options to determine whether classical (normal), rating-aware (rating), rank-aware (rank), or rank aware
            with positional discounts (rank_pos) SOM is performed
        alpha: float, optional
            Hyperparameter to tune importance of ranking vs. local similarity. 
        """
        # default values
        n = 5
        lr = 1
        sigma = 1
        max_iter = 10000
        epochs = 10

        #print(type(parameters),parameters)
        #TODO check for empty parameters
        if type(self._parameters) is dict:
            # Modify default values if parameters present
            if "n" in self._parameters:
                n = self._parameters["n"]
            if "lr" in self._parameters:
                lr = self._parameters["lr"]
            if "sigma" in self._parameters:
                sigma = self._parameters["sigma"]
            if "max_iter" in self._parameters:
                max_iter = self._parameters["max_iter"]
            if "epochs" in self._parameters:
                epochs = self._parameters["epochs"]

        
        self.n = n
        m = self.n #simplification to consider only squared SOMs
        self.m = m

        self.dim = data.shape[0]
        dim = self.dim
        self.shape = (n, n)
        self.initial_lr = lr
        self.lr = lr
        self.sigma = sigma
        self.max_iter = max_iter

        self.epochs = epochs
        
        #initialize rankingSOM specific features
        self.som_type = "normal"
        self.alpha = 1


        # Initialize weights
        self.random_state = 42
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(size=(m * n, dim))
        self._locations = self._get_locations(m, n)

        # Set after fitting
        self._inertia = None
        self._n_iter_ = None
        self._trained = False


        """
        Take data (a tensor of type float64) as input and fit the SOM to that
        data for the specified number of epochs.
        Parameters
        ----------
        X : ndarray
            Training data. Must have shape (n, self.dim) where n is the number
            of training samples.
        ranks: ndarray
            Ranks of the training data w.r.t. original ordering
        ratings: ndarray
            Ratings of the training data w.r.t. original ordering
        epochs : int, default=1
            The number of times to loop through the training data when fitting.
        shuffle : bool, default True
            Whether or not to randomize the order of train data when fitting.
            Can be seeded with np.random.seed() prior to calling fit.
        Returns
        -------
        None
            Fits the SOM to the given data but does not return anything.
        """

        X = data.T #transpose the data => our objects are features 
        #TODO transform nominal data to numeric

        #print(X.shape)
        #topK = np.argsort(ranks)[:(self.m*self.n)]
        #self.expected_ratings = ratings[topK]
        self.volume_of_samples = len(X)
        #print(self.volume_of_samples)

        # Count total number of iterations
        global_iter_counter = 0
        n_samples = X.shape[0]
        total_iterations = np.minimum(epochs * n_samples, self.max_iter)
        
        #prepare for probabilistic candidates selection
        #rngForCandidateSelection = np.random.default_rng(self.random_state)
        #prob = ratings/np.sum(ratings)
        
        for epoch in range(epochs):
            # Break if past max number of iterations
            if global_iter_counter > self.max_iter:
                break
            
            #if self.use_prob_candidate_selection:
            #    indices = rngForCandidateSelection.choice(np.arange(n_samples) , size=n_samples, p=prob)             
            #elif shuffle:
            rng = np.random.default_rng(self.random_state)
            indices = rng.permutation(n_samples)
            #print(indices)
            #else:
            #    indices = np.arange(n_samples)

            # Train
            for idx in indices:
                # Break if past max number of iterations
                if global_iter_counter > self.max_iter:
                    break
                input = X.iloc[idx]
                #print(input.shape)
                #rank = ranks[idx]
                #rating = ratings[idx]
                # Do one step of training
                self.step(input)#, rank, rating)
                # Update learning rate
                global_iter_counter += 1
                self.lr = (1 - (global_iter_counter / total_iterations)) * self.initial_lr

        # Compute inertia
        inertia = np.sum(np.array([float(self._compute_point_intertia(X.iloc[idx])) for idx in range(X.shape[0])]))
        self._inertia_ = inertia

        # Set n_iter_ attribute
        self._n_iter_ = global_iter_counter

        # Set trained flag
        self._trained = True

        # store trained positions of features
        self._feature_lin_labels = self._predict(X)
        #print(self._feature_lin_labels)
        #self._feature_coords = np.unravel_index(self._feature_lin_labels,(self.n,self.m))
        self._feature_coords = self._locations[self._feature_lin_labels,:]
        #print(self._feature_coords)

        self._train_normalization(data)
        self._train_mean_image(data)
        return     



    def _predict(self, X): #, ranks, ratings):
        """
        Predict BMU for each element in X (i.e., predict location for each feature in the original dataframe).
        Parameters
        ----------
        X : ndarray
            An ndarray of shape (n, self.dim) where n is the number of samples.
            The data to predict clusters for.
        Returns
        -------
        labels : ndarray
            An ndarray of shape (n,). The predicted cluster index for each item
            in X.
        """
        # Check to make sure SOM has been fit
        if not self._trained:
            raise NotImplementedError('SOM object has no predict() method until after calling fit().')

        # Make sure X has proper shape
        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'
        assert X.shape[1] == self.dim, f'This SOM has dimesnion {self.dim}. Received input with dimension {X.shape[1]}'
        
        
        #indices = range(len(ranks))
        labels = np.array([self._find_bmu(X.iloc[idx]) for idx in range(X.shape[0])])
        #if self.som_type == "normal":
        #    labels = np.array([self._find_bmu(X[idx,:]) for idx in indices])
        #elif self.som_type == "rank":
        #    labels = np.array([self._find_bmu_rank_aware(X[idx,:], ranks[idx], ratings[idx]) for idx in indices])

        #TODO normalize values

        return labels       
        

    def _find_bmu(self, x):
        """
        Find the index of the best matching unit for the input vector x.
        """
        # Stack x to have one row per weight
        x_stack = np.stack([x]*(self.m*self.n), axis=0)
        # Calculate distance between x and each weight
        distance = np.linalg.norm(x_stack - self.weights, axis=1)
        # Find index of best matching unit

        #print(distance.shape, self.weights.shape)

        return np.argmin(distance)

    def _get_locations(self, m, n):
        """
        Return the indices of an m by n array.
        """
        return np.argwhere(np.ones(shape=(m, n))).astype(np.int64)

    def step(self, x):#, rank, rating):
        """
        Do one step of training on the given input vector.
        """
        # Stack x to have one row per weight
        x_stack = np.stack([x]*(self.m*self.n), axis=0)

        # Get index of best matching unit
        # TODO: Based on the SOM type, different procedures are applied here
        # - this is the only difference between normal SOM and rank-aware SOM
        
        """
        #classical (normal), rating-aware (rating), rank-aware (rank), or rank aware with positional discounts (rank_pos)
        if self.som_type == "normal":
            bmu_index = self._find_bmu(x)
        elif self.som_type == "rating":
            bmu_index = self._find_bmu_rating_aware(x, rank, rating)        
        elif self.som_type == "rank":
            bmu_index = self._find_bmu_rank_aware(x, rank, rating)
        elif self.som_type == "rank_pos":
            bmu_index = self._find_bmu_rank_aware_positional(x, rank, rating)            
        """
        
        bmu_index = self._find_bmu(x)        
        # Find location of best matching unit
        bmu_location = self._locations[bmu_index,:]

        

        # Find square distance from each weight to the BMU
        stacked_bmu = np.stack([bmu_location]*(self.m*self.n), axis=0)
        bmu_distance = np.sum(np.power(self._locations.astype(np.float64) - stacked_bmu.astype(np.float64), 2), axis=1)

        # Compute update neighborhood
        neighborhood = np.exp((bmu_distance / (self.sigma ** 2)) * -1)

        local_step = self.lr * neighborhood

        # Stack local step to be proper shape for update
        local_multiplier = np.stack([local_step]*(self.dim), axis=1)

        # Multiply by difference between input and weights
        delta = local_multiplier * (x_stack - self.weights)

        # Update weights
        self.weights += delta

    def _compute_point_intertia(self, x):
        """
        Compute the inertia of a single point. Inertia defined as squared distance
        from point to closest cluster center (BMU)
        """
        #print(x.shape)
        # Find BMU
        bmu_index = self._find_bmu(x)
        bmu = self.weights[bmu_index]
        # Compute sum of squared distance (just euclidean distance) from x to bmu
        return np.sum(np.square(x - bmu))

    """
    return a DF of coordinates and corresponding values (RGB or greyscale) for individual objects
    this should be changed if we want some other aggregation type (e.g. max instead of mean)
    """
    def _calculate_image_grid(self,featureVector):
        df = pd.DataFrame({"ids":[0]*len(featureVector), "keys":self._feature_lin_labels,"vals": featureVector})
        df = df.groupby(["ids","keys"]).sum().reset_index()

        pivot = pd.pivot_table(df, values="vals", index="ids", columns="keys")

        transformedPivot = self.dt_transformer.transform(pivot)

        df = pd.DataFrame({"keys":pivot.columns,"vals": transformedPivot.reshape(-1)})

        #print(df)
        return df
    
    def _train_normalization(self, data):
        self.dt_transformer = QuantileTransformer(random_state=42)

        vals = data.values.reshape(-1)
        keys = np.array(list(self._feature_lin_labels)*data.shape[0])
        ids = np.array([[i]*data.shape[1] for i in range(data.shape[0])]).reshape(-1)
        #print(vals.shape, keys.shape, ids.shape)
        df = pd.DataFrame({"ids":ids, "keys":keys, "vals":vals})
        df = df.groupby(["ids","keys"]).sum().reset_index()
        pivot = pd.pivot_table(df, values="vals", index="ids", columns="keys")
        self._data_pivot = pivot
        #print(pivot)

        self.dt_transformer.fit(pivot)

        return     

    def _train_mean_image(self, data: pd.DataFrame):
        #data: dataframe of all images
        coordsDF = pd.DataFrame({"keys":[],"vals": []})

        for idx, row in data.iterrows():
            coordsDF = coordsDF.append(self._calculate_image_grid(row))

        coordsDF = coordsDF.groupby("keys").mean().reset_index()

        #print(coordsDF)
        self._coordsDF = coordsDF
        return


    def _displayOneElement(self, ax, x, y, color):
        rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='none', facecolor=color)
        ax.add_patch(rect)


    def _finalizeImage(self,ax, title, n):
        ax.set_ylim(0, n)
        ax.set_xlim(0, n)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.set_title(title) 
           


    def transform_one(self, data: pd.Series, output: str) -> None:
        #print(data.shape)
        coordsDF = self._calculate_image_grid(data)
        #print(coordsDF)

        fig,ax = plt.subplots(1,1, figsize=(4, 4))

        for idx,row in coordsDF.iterrows():
            pos = self._locations[idx,:]
            #print(pos)
            coordMean = self._coordsDF.loc[idx,"vals"]
            epsilon = 10e-6
            if row["vals"] - coordMean >=0 : #red area
                blueCol = 1 - ((row["vals"] - coordMean) / (1 - coordMean + epsilon))
                greenCol = blueCol
                redCol = 1
            else:
                redCol =  (coordMean - row["vals"]) / (coordMean + epsilon)
                greenCol = redCol
                blueCol = 1

            self._displayOneElement(ax,pos[0],pos[1],[redCol,greenCol,blueCol,1])
        
        self._finalizeImage(ax,"",self.n)

        ax.axis('off')
        plt.savefig(output, pad_inches=0, bbox_inches='tight', transparent=False)
        plt.close(fig)

    def transform(self, data: pd.DataFrame, labels: pd.DataFrame, output: str) -> None:
        # if self._coords is None:
        #     raise SyntaxError('Method ::fit must called before ::transform.')
        if self._ordering is not None:
            data = data.iloc[:, self._ordering]
        super().transform(data, labels, output)
