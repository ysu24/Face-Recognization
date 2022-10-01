'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
Yiheng Su
CS 252 Data Analysis Visualization
Spring 2022
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset.

        (No changes should be needed)
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

        self.normalized_info = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`
        '''
        # center the data
        centered_data = data - data.mean(axis=0)
        cov_matrix = centered_data.T @ centered_data / (centered_data.shape[0]-1)
        return cov_matrix

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        sorted_e_vals = np.sort(e_vals)[::-1]
        e_vals_sum = np.sum(e_vals)
        proportion = [eval / e_vals_sum for eval in sorted_e_vals]
        return proportion

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        cum_sum = []
        for i in range(len(prop_var)):
            current_sum = prop_var[0]
            for j in range(1, i+1):
                current_sum += prop_var[j]
            cum_sum.append(current_sum)
        
        return cum_sum
                

    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
            - Before normalizing, create instance variables containing information that would be
            needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''
        self.vars = vars
        self.A = np.array(self.data[vars])
        self.normalized = normalize

        if normalize:
            mins = self.A.min(axis=0)
            maxs = self.A.max(axis=0)
            self.normalized_info = [mins, maxs - mins]
            self.A = (self.A - mins) / (maxs - mins)
#             self.normalized = self.A.copy()

        cov_mat = self.covariance_matrix(self.A)
        (self.e_vals, self.e_vecs) = np.linalg.eig(cov_mat)

        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)

        
    def scree_plot(self):
        plt.figure(figsize=(12,8))

        plt.bar(range(len(self.prop_var)), self.prop_var, alpha=0.5, align='center',
                label='Proportional variance')

        plt.ylabel('Proportional variance ratio')
        plt.xlabel('Ranked Principal Components')
        plt.title("Scree Graph")
        plt.ylim(0, np.max(self.prop_var))

        plt.legend(loc='best')
       

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        if num_pcs_to_keep is None:
            var=self.cum_var
        else:
            var=self.cum_var[:num_pcs_to_keep]
        x=np.arange(len(var)+1)
        var=np.real(np.hstack(([0.0], var)))
        plt.figure(figsize=(12,8))
        plt.plot(x, var, 'o', markersize=2)
        plt.plot(x, var, '-', color='pink')
        plt.xlim(-len(var)/40,None)
        plt.xlabel("PCs", fontsize=15)
        plt.ylabel("accumulated variance", fontsize=15)
        if num_pcs_to_keep is None:
            plt.title(f"Elbow Plot of Top all PCs", fontsize=20)
        else:
            plt.title(f"Elbow Plot of Top {num_pcs_to_keep} PCs", fontsize=20)


    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''
        proj_vecs = self.e_vecs[:,pcs_to_keep]
        self.A_proj = self.A @ proj_vecs
        
        return self.A_proj
        

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        - If you normalized, remember to rescale the data projected back to the original data space.
        '''
        proj_vecs = self.e_vecs[:, :top_k]
        self.A_proj = self.pca_project(np.arange(top_k))

        if self.normalized:
            mean = self.A.mean(axis=0)
            data_back = self.A_proj @ proj_vecs.T + mean
            data_back = data_back * self.normalized_info[1] + self.normalized_info[0]
            return data_back
        
        return self.A_proj @ proj_vecs.T
    
    def face_recognition(self, query_face, top_k, tolerance_level=10): 
        e_vecs_proj = self.e_vecs[:, :top_k]
        a_proj=self.pca_project(top_k)
        qu_im_proj=query_face.reshape(1, 64*64) @ e_vecs_proj
        for i in range(len(self.A)):
            img=self.A[i][np.newaxis, :]
            vec = img @ e_vecs_proj
            if self.euclidean_distance(vec, qu_im_proj) <= tolerance_level:
                print("Min distance:", self.euclidean_distance(vec, qu_im_proj))
                return i
        
        return -1
        
    
    def euclidean_distance(self, vec1, vec2):
        vec = vec1 - vec2
        return np.sqrt(np.sum(np.square(vec)))

