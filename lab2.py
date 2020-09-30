""" CS4243 Lab 2: Image Segmentations
See accompanying Jupyter notebook (lab2.ipynb) and PDF (lab2.pdf) for instructions.
"""
import cv2
import numpy as np
import random

from time import time
from skimage import color
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed


# Part 1 

def smoothing(img):
    """Smooth image using Guassain filter.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).

    Returns:
        img_smoothed (np.ndarray)   : Output smoothed image of size (H, W, 3).

    """

    """ YOUR CODE STARTS HERE """
    img_smoothed = cv2.GaussianBlur(img,(5,5),5.0)

    """ YOUR CODE ENDS HERE """

    return img_smoothed

def RGBtoLab(img):
    """Convert RGB image into L*a*b color space.

    Args:
        img (np.ndarray)            : Input RGB image  of size (H, W, 3).


    Returns:
        lab (np.ndarray)            : Converted L*a*b image of size (H, W, 3).

    """

    """ YOUR CODE STARTS HERE """
    lab = color.rgb2lab(img)

    """ YOUR CODE ENDS HERE """
   
    return lab



# Part 2
def k_means_clustering(data,k):
    """ Estimate clustering centers using k-means algorithm.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        k (int)                     : Number of centroids

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """
    start = time()


    """ YOUR CODE STARTS HERE """
    num = data.shape[0]
    dim = data.shape[1]
    
    import math
    centers = data[random.sample(range(num), k)]
    
    labels = np.zeros(num)

    change = 1
    
    while change > 1e-10:
        change = 0
        """ Get the labels (centres) for each of the datapoints """
        for i in range(num):
            dist = []
            for j in range(k):
                dist.append(math.sqrt(sum(pow(centers[j, :] - data[i, :], 2))))
            minCluserId = np.argmin(dist)
            labels[i] = minCluserId
        
        tempCenters = np.zeros((k, dim))
        tempMean = np.zeros((k, dim))
        
        
        for j in range(k):
            """ For each centers that we currently have, find the mean of the cluster"""
            tempNums = 0
            for i in range(num):
                """ if the data belongs to this center j, then we add the features to find the sum"""
                if j == labels[i]:
                    tempMean[j] = tempMean[j] + data[i]
                    tempNums = tempNums + 1

            if (tempNums == 0): 
                break
            else:
                tempMean[j] = tempMean[j]/tempNums
            
            check = 0
            curr_min_value = np.full((dim), 1000000)
            """ For each centers that we currently have, find the new center """
            for c in range (num):
                if j == labels[c]:
                    if (check == 0):
                        curr_min_value = np.sum(np.abs(data[c] - tempMean[j]))
                        check = 1
                    diff = data[c] - tempMean[j]
                    if (np.sum(np.abs(diff)) < np.sum(curr_min_value)):
                        curr_min_value = np.abs(diff)
                        tempCenters[j] = data[c] 
                        
        change = np.sum(np.absolute(centers - tempCenters))
        centers = tempCenters

    """ YOUR CODE ENDS HERE """


    end =  time()
    kmeans_runtime = end - start
    print("K-means running time: %.3fs."% kmeans_runtime)
    return labels, centers



def get_bin_seeds(data, bin_size, min_bin_freq=1):
    """ Generate initial bin seeds for windows sampling.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        bin_size (float)            : Bandwidth.
        min_bin_freq (int)          : For each bin_seed, number of the minimal points should cover.

    Returns:
        bin_seeds (List)            : Reprojected bin seeds. All bin seeds with total point number 
                                      bigger than the threshold.
    """

    """ YOUR CODE STARTS HERE """
    compress_coordinate = np.vectorize(lambda x: np.round(x/bin_size))
    compressed = compress_coordinate(data)
    
    group = np.unique(compressed, axis = 0, return_counts = True)
    old_bin_seeds = group[0]
    old_freqs = group[1]
    
    new_bin_seeds = []
    new_freqs = []
    for i in range(len(old_freqs)):
        if old_freqs[i] >= min_bin_freq:
            new_bin_seeds.append(old_bin_seeds[i])
            new_freqs.append(old_freqs[i])
            
    bin_seeds = [u_m * bin_size for u_m in new_bin_seeds]
    
    """ YOUR CODE ENDS HERE """
    return bin_seeds

def mean_shift_single_seed(start_seed, data, nbrs, max_iter):
    """ Find mean-shift peak for given starting point.

    Args:
        start_seed (np.ndarray)     : Coordinate (x, y) of start seed. 
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        nbrs (class)                : Class sklearn.neighbors._unsupervised.NearestNeighbors.
        max_iter (int)              : Max iteration for mean shift.

    Returns:
        peak (tuple)                : Coordinate (x,y) of peak(center) of the attraction basin.
        n_points (int)              : Number of points in the attraction basin.
                              
    """

    # For each seed, climb gradient until convergence or max_iter
    bandwidth = nbrs.get_params()['radius']
    stop_thresh = 1e-3 * bandwidth  # when mean has converged

    """ YOUR CODE STARTS HERE """
    iter_no = 0
    shift = 0
    start_mean = start_seed
    
    while iter_no < max_iter and shift < stop_thresh:
        iter_no += 1

        # Find the nearest points
        nbrs.fit(data)
        points_nearest_to_seed = nbrs.radius_neighbors(start_mean.reshape(1, -1))[1][0]

        new_mean = np.zeros((1, 2))
        no_of_points = 0

        # Find out the new mean from the current neighbours
        for index in points_nearest_to_seed:
            curr_point = data[index]
            new_mean[0][0] += curr_point[0]
            new_mean[0][1] += curr_point[1]
            no_of_points += 1
        new_mean[0][0] /= no_of_points
        new_mean[0][1] /= no_of_points
        
        # Check if converges using Euclidean distance
        shift = np.linalg.norm(start_mean-new_mean)
        start_mean = new_mean
        
    peak = tuple(start_mean[0])
    n_points = no_of_points
    
    """ YOUR CODE ENDS HERE """

    return peak, n_points


def mean_shift_clustering(data, bandwidth=0.7, min_bin_freq=5, max_iter=300):
    """pipline of mean shift clustering.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        bandwidth (float)           : Bandwidth parameter for mean shift algorithm.
        min_bin_freq(int)           : Parameter for get_bin_seeds function.
                                      For each bin_seed, number of the minimal points should cover.
        max_iter (int)              : Max iteration for mean shift.

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)
    """
    start = time()
    n_jobs = None
    seeds = get_bin_seeds(data, bandwidth, min_bin_freq)
    n_samples, n_features = data.shape
    center_intensity_dict = {}

    # We use n_jobs=1 because this will be used in nested calls under
    # parallel calls to _mean_shift_single_seed so there is no need for
    # for further parallelism.
    nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(data)
    # execute iterations on all seeds in parallel
    all_res = Parallel(n_jobs=n_jobs)(
        delayed(mean_shift_single_seed)
        (seed, data, nbrs, max_iter) for seed in seeds)

    # copy results in a dictionary
    for i in range(len(seeds)):
        if all_res[i] is not None:
            center_intensity_dict[all_res[i][0]] = all_res[i][1]

    if not center_intensity_dict:
        # nothing near seeds
        raise ValueError("No point was within bandwidth=%f of any seed."
                         " Try a different seeding strategy \
                         or increase the bandwidth."
                         % bandwidth)
    


    """ YOUR CODE STARTS HERE """
    sort_intensity_dict = sorted(center_intensity_dict.items(), key=lambda x: x[1], reverse=True)
    list_of_peaks = []  # Sort peaks based on descending frequency
    for tup in sort_intensity_dict:
        list_of_peaks.append(tup[0])
    new_nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(list_of_peaks)

    # Find groups of near duplicate peaks
    new_grouping = []
    for point in list_of_peaks:
        near_duplicate_peak = new_nbrs.radius_neighbors(np.asarray(point).reshape(1, -1))[1][0]
        near_duplicate_peak = near_duplicate_peak.tolist()
        if near_duplicate_peak not in new_grouping:
            new_grouping.append(near_duplicate_peak)

    # Clean up new grouping to remove duplicates: Necessary for case 3
    for i in range(len(new_grouping)):
        group = new_grouping[i]
        for ele in group:      
            for j in range(i+1, len(new_grouping)):
                if ele in new_grouping[j]:
                    new_grouping[j].remove(ele)
    
    new_grouping = [x for x in new_grouping if x]  # Remove empty lists
    k = len(new_grouping)
    centers = np.zeros((k, n_features))
    
    # Choose the peak with greatest number of points
    for k in range(k):
        grouping = new_grouping[k]
        max_freq = 0
        max_index = -1
        for index in grouping:
            peak = list_of_peaks[index]
            freq = center_intensity_dict[peak]
            if freq > max_freq:
                max_freq = freq
                max_index = index
        centers[k] = list_of_peaks[max_index]
        
    # Iterate and assign to nearest cluster peak
    final_nbrs = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(centers)
    labels = final_nbrs.kneighbors(data)[1]

    """ YOUR CODE ENDS HERE """
    end =  time()
    kmeans_runtime = end - start
    print("mean shift running time: %.3fs."% kmeans_runtime)
    return labels, centers









#Part 3:

def k_means_segmentation(img, k):
    """Descrption.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).
        k (int)                     : Number of centroids
    
    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """

    """ YOUR CODE STARTS HERE """
    img_dim = img.shape
    
    if len(img_dim) == 2:
        Height = img_dim[0]
        Width = img_dim[1]
        Feature = 1
    elif len(img_dim) == 3:
        Height = img_dim[0]
        Width = img_dim[1]
        Feature = img_dim[2]
        
    labels, centers = k_means_clustering(img.reshape(Height * Width, Feature), k)
    """ YOUR CODE ENDS HERE """

    return labels,centers


def mean_shift_segmentation(img,b):
    """Descrption.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).
        b (float)                     : Bandwidth.

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """

    """ YOUR CODE STARTS HERE """
    img_dim = img.shape
    
    if len(img_dim) == 2:
        Height = img_dim[0]
        Width = img_dim[1]
        Feature = 1
    elif len(img_dim) == 3:
        Height = img_dim[0]
        Width = img_dim[1]
        Feature = img_dim[2]
    
    labels, centers = mean_shift_clustering(img.reshape(Height * Width, Feature), b)
    
    """ YOUR CODE ENDS HERE """

    return labels, centers














"""Helper functions: You should not have to touch the following functions.
"""
def load_image(im_path):
    """Loads image and converts to RGB format

    Args:
        im_path (str): Path to image

    Returns:
        im (np.ndarray): Loaded image (H, W, 3), of type np.uint8.
    """


    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def colors(k):

    """generate the color for the plt.scatter.

    Args:
        k (int): the number of the centroids

    Returns:
        ret (list): list of colors .

    """

    colour = ["coral", "dodgerblue", "limegreen", "deeppink", "orange", "darkcyan", "rosybrown", "lightskyblue", "navy"]
    if k <= len(colour):
        ret = colour[0:k]
    else:
        ret = []
        for i in range(k):
            ret.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
    return ret

def stack_seg(img, labels, centers):
    """stack segmentations for visualization.

    Args:
        img (np.ndarray): image
        labels(np.ndarray): lables for every pixel. 
        centers(np.ndarray): cluster centers.

    Returns:
        np.vstack(result) (np.ndarray): stacked result.

    """

    labels = labels.reshape((img.shape[:-1]))
    reduced = np.uint8(centers)[labels]
    result = [np.hstack([img])]
    for i, c in enumerate(centers):
        mask = cv2.inRange(labels, i, i)
        mask = np.dstack([mask]*3) # Make it 3 channel
        ex_img = cv2.bitwise_and(img, mask)
        result.append(np.hstack([ex_img]))

    return np.vstack(result)
