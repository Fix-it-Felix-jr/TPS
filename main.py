import matplotlib.pyplot as plt
from sqlalchemy import Float
import numpy as np
import torch
import pyvista
from os.path import join
import csv
import argparse
from tps_registration.registration import registration
from tps_registration.tools import helmert_mat
import os



parser = argparse.ArgumentParser()
#-s STEPS -f FILENAME -d DIMENSIONS -l NUMBER_OF_LANDMARKS_PER_SHAPE
parser.add_argument("-s", "--steps", help="Nunber of steps", type=int)
parser.add_argument("-d", "--dimensions", help="Number of dimensions", type=int)
parser.add_argument("-l", "--landmarks", help="Number of landmarks per shape", type=int)
parser.add_argument("-m", "--mu", help="Mu value", type=float)
args = parser.parse_args()



def centra(matrice,n_steps):
    for i in range(n_steps):
        mat=matrice[i]
        c=[0.0,0.0]
        for p in range(0,mat.size()[0]):
            c[0]+=mat[p][0]
            c[1]+=mat[p][1]

        c[0]/=mat.size()[0]
        c[1]/=mat.size()[1]

        for p in range(0,mat.size()[0]):
            #centro la figura
            matrice[i][p][0]-=c[0]
            matrice[i][p][1]-=c[1]

        #Ricalcolo le decentrature
        c=[0.0,0.0]
        for p in range(0,mat.size()[0]):
            c[0]+=mat[p][0]
            c[1]+=mat[p][1]

        c[0]/=mat.size()[0]
        c[1]/=mat.size()[1]

        #print("------------------------")
        #print("Print del centroide:",c)
    return matrice

def findCentro(matrice,n_steps):
    for i in range(n_steps):
        mat=matrice[i]
        c=[0.0,0.0]
        for p in range(0,mat.size()[0]):
            c[0]+=mat[p][0]
            c[1]+=mat[p][1]

        c[0]/=mat.size()[0]
        c[1]/=mat.size()[1]

        print("------------------------")
        print("Print del centroide:",c)

def plot_2d(trajectory, data):
    lim = data.max() * 1.2
    fig = plt.figure(figsize=(12, 12))
    for i in range(n_steps):
        ax = plt.subplot(3, 3, i + 1)

        mesh = np.concatenate([data[i], data[i, 0, None]])
        plt.plot(mesh[:, 0], mesh[:, 1], 'o-', color='black', markersize=2)
        pt_i = trajectory[i]
        pt = np.concatenate([pt_i, pt_i[0, None]])
        plt.plot(pt[:, 0], pt[:, 1], 'o-', color='r', markersize=2)

        plt.xlim([-lim, lim])
        plt.ylim([-lim, lim])

    for ax_ in fig.axes:
        ax_.set_aspect('equal')
    plt.title(f'{deformation}')
    plt.show()

def saveToFile(output):
    input_variable = output
    with open('Output.csv', 'w', newline = '') as csvfile:
        my_writer = csv.writer(csvfile, delimiter = ' ')
        for i in range(0,(n_steps-1)):
            for p in range(0,(lands-1)):
                my_writer.writerow(trajectory[i][p].numpy().tolist())

def plot_3d(trajectory):
    for j, points in enumerate(trajectory):
        connections = pyvista.read(diastole)
        connections.points = points.numpy()
        connections.save(join(data_dir, 'tps', f'time_{j}.vtk'))

if __name__ == '__main__':
    
    print("The input file has to be saved in the data folder, and its name has to be input.txt")
    
    n_steps = args.steps #numero di steps
    deformation = 'input' #nome del file dove sono contenute le forme
    dim=args.dimensions #dimensioni della forma che viene data in input
    lands=args.landmarks #numero di landmarks per ogni forma
    mu=args.mu #numero di landmarks per ogni forma
    rhos = {
        'Bending+affine': 2.71, 'Bending+size': 2.51, 'Pure bending': 2.51, 'Pure affine': 2.77,
        'Pure scaling': 2.59, 'input': 2.51
    }

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data')
    #codice 3D
    if dim==3:
        heart_dir = os.path.join(dirname, 'data/input.txt')
        originals = torch.from_numpy(np.loadtxt(heart_dir)).reshape(n_steps, lands, dim)
        diastole, systole = originals
        trajectory = registration(diastole, systole, n_steps, data_dir, dim=3, lr=1.e-2, max_iter=100, muvalue=mu)
        saveToFile(trajectory) #salvo l'output in Output.csv
        plot_3d(trajectory)

    #codice 2D
    if dim==2:
        file = join(data_dir, f'{deformation}.txt')
        originals = np.loadtxt(file).reshape((n_steps, lands, dim))
        landmarks = torch.from_numpy(originals)
        trajectory = registration(
            landmarks[0], landmarks[-1], n_steps, data_dir, lr=1.e-3, max_iter=1150,
            rho=rhos[deformation], kappa_1=5., kappa_2=100., muvalue=mu
        )

        output=centra(trajectory,n_steps)
        saveToFile(output) #salvo l'output in Output.csv
        plot_2d(output, centra(torch.tensor(originals),n_steps))   

    findCentro(output,n_steps) #Verifica della centratura

    np.savetxt(join(data_dir, f'tps_output_{deformation}.txt'), output.reshape(n_steps, -1))
    plt.savefig(join(data_dir, 'images', f'tps_{deformation}.svg'))
