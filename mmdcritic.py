"""
MMD-critic (https://papers.nips.cc/paper/2016/hash/5680522b8e2bb01943234bce7bf84534-Abstract.html)
Copyright (C) 2016 Been Kim <beenkim@csail.mit.edu>
Copyright (C) 2016 Rajiv Khanna <rajivak@utexas.edu>
Copyright (C) 2016 Oluwasanmi Koyejo <sanmi@illinois.edu>
mmdcritic.py
Copyright (C) 2021 Elodie Escriva, Kaduceo <elodie.escriva@kaduceo.com>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

# Global kernel
def calculate_kernel(X, g=None):
    return rbf_kernel(X, gamma=g)


# Local kernel
def calculate_kernel_individual(X, y, g=None):
    kernel = np.zeros((np.shape(X)[0], np.shape(X)[0]))
    # sortind = np.argsort(y).values
    # sortind = np.argsort(y)
    # print(sortind)
    X=pd.DataFrame(X)
    y=pd.Series(y)
    sortind = y.argsort()
    X = X.loc[sortind, :]
    y = y.loc[sortind]

    for i in np.arange(y.nunique()):
        ind = np.where(y == i)[0]
        startind = np.min(ind)
        endind = np.max(ind) + 1
        kernel[startind:endind, startind:endind] = rbf_kernel(
            X.iloc[startind:endind, :], gamma=g
        )
    return kernel


def greedy_select_protos(K, candidate_indices, m):
    """
    Select prototypes based on the kernel matrix.
    Parameters
    ----------
    K : 2-dimension numpy array
        Kernel matrix to use to select prototypes.
    candidate_indices : numpy array
        Array of the instances indices.
    m : int
        Number of prototypes to select.
    Returns
    -------
    Numpy Array
        Array of indices of the selected instances.
    """

    if len(candidate_indices) != np.shape(K)[0]:
        K = K[:, candidate_indices][candidate_indices, :]

    n = len(candidate_indices)

    colsum = 2 * np.sum(K, axis=0) / n

    selected = np.array([], dtype=int)
    for i in range(m):
        argmax = -1
        candidates = np.setdiff1d(range(n), selected)

        s1array = colsum[candidates]
        if len(selected) > 0:
            temp = K[selected, :][:, candidates]
            s2array = np.sum(temp, axis=0) * 2 + np.diagonal(K)[candidates]
            s2array /= len(selected) + 1
            s1array -= s2array
        else:
            s1array -= np.abs(np.diagonal(K)[candidates])

        argmax = candidates[np.argmax(s1array)]
        selected = np.append(selected, argmax)

    return candidate_indices[selected]


def select_criticism_regularized(K, selectedprotos, m, reg="logdet"):
    """
    Selects criticicms based on the previously selected prototypes and the input kernel matrix.
    Parameters
    ----------
    K : 2-dimension numpy array
        Kernel matrix to use to select criticisms.
    selectedprotos : Numpy Array
        Indices of the instances selected as prototypes.
    m : int
        Number of criticisms to select.
    reg : string, optional
        Name of the regulizer to use. Default "logdet".
    Returns
    -------
    selected : Numpy Array
        Array of indices of the selected instances.
    """

    n = np.shape(K)[0]
    available_reg = {None, "logdet", "iterative"}
    assert (
        reg in available_reg
    ), f'Unknown regularizer: "{reg}". Available regularizers: {available_reg}'

    selected = np.array([], dtype=int)
    candidates2 = np.setdiff1d(range(n), selectedprotos)
    inverse_of_prev_selected = None  # should be a matrix

    colsum = np.sum(K, axis=0) / n
    inverse_of_prev_selected = None

    for i in range(m):
        argmax = -1
        candidates = np.setdiff1d(candidates2, selected)

        s1array = colsum[candidates]

        temp = K[selectedprotos, :][:, candidates]
        s2array = np.sum(temp, axis=0)
        s2array /= len(selectedprotos)
        s1array -= s2array

        if reg == "logdet":
            if inverse_of_prev_selected is not None:  # first call has been made already
                temp = K[selected, :][:, candidates]

                # hadamard product
                temp2 = np.array(np.dot(inverse_of_prev_selected, temp))
                regularizer = temp2 * temp
                regcolsum = np.sum(regularizer, axis=0)
                regularizer = np.log(abs(np.diagonal(K)[candidates] - regcolsum))
                s1array += regularizer
            else:
                s1array -= np.log(abs(np.diagonal(K)[candidates]))

        argmax = candidates[np.argmax(s1array)]
        selected = np.append(selected, argmax)

        KK = K[selected, :][:, selected]
        inverse_of_prev_selected = np.linalg.pinv(KK)  # shortcut

        if reg == "iterative":
            selectedprotos = np.append(selectedprotos, argmax)

    return selected


def mmd_critic(X, y, n, m, gamma=None, ktype=0, crit=True):
    """
    Selects the prototypes and criticisms instances from the input dataset.
    
    Parameters
    ----------
    X : Pandas DataFrame
        Input dataset : Raw datas or influences.
    y : Pandas DataFrame
        Labels of the instances in the input dataset.
    n : int
        Number of prototypes desired.
    m : int
        Number of criticisms desired.
    gamma : float, optional
        parameter for the kernel exp(- gamma * \| x1 - x2 \|_2). Default None.
    ktype : bool, optional
        Type of kernel to use. 0 for global, 1 for local. Default 0.
    crit : bool, optional
        Flag to compute or not criticisms. Default True.
    Returns
    -------
    Numpy Array
        Array of the selected prototypes and criticism instances indices.
    """
    if ktype == 0:
        kernel = calculate_kernel(X, gamma)
    else:
        kernel = calculate_kernel_individual(X, y, gamma)

    selected = greedy_select_protos(kernel, np.array(range(np.shape(kernel)[0])), n)

    critselected = np.array([], dtype=int)
    if crit:
        critselected = select_criticism_regularized(kernel, selected, m, reg="logdet")

    return np.concatenate((selected, critselected))