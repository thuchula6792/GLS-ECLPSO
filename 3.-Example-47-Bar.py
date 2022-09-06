# ***************************************************************************************#
""" Author:  Soviphou Muong, Ph.D Thu Huynh Van, Assoc. Prof. Sawekchai Tangaramvong 
#   Emails:  thuxd11@gmail.com, Sawekchai.T@chula.ac.th
#            Applied Mechanics and Structures Research Unit, Department of Civil Engineering, 
#            Chulalongkorn University 
#   https://scholar.google.com/citations?user=NysMfoAAAAAJ&hl=vi 
# Research paper: Combined Enhanced Comprehensive Learning PSO and Gaussian Local
# Search for Sizing and Shape Optimization of Truss Structures (2022) "Building"
""" 
# Reference: Thu Huynh Van, Sawekchai Tangaramvong (2022). Two-Phase ESO-CLPSO Method for the Optimal Design 
# of Structures with Discrete Steel Sections. "Advances in Engineering Software". https://doi.org/10.1016/j.advengsoft.2022.103102
# CLPSO code: https://github.com/thuchula6792/CLPSO

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import copy
import os
import random
import pandas as pd

# TRUSS STRUCTURE
class Truss:
    def __init__(self, young_modulus, density, truss_system, node, bar):
        self.young_modulus = young_modulus
        self.density = density
        self.truss_system = truss_system
        if self.truss_system == '2d':
            self.dof = 2
        elif self.truss_system == '3d':
            self.dof = 3
        else:
            raise ValueError('Truss system is either 2d or 3d.')
        self.node = node.astype(float)
        self.bar = bar.astype(int)
        self.load = np.zeros_like(node)
        self.support = np.ones_like(node).astype(int)
        self.section = np.ones(len(bar))
        self.force = np.array(len(bar))
        self.stress = np.array(len(bar))
        self.buckling = np.array(len(bar))
        self.displacement = np.array(len(node))
        self.weight = 0

    def analysis(self, buckling_factor=None):
        nn = len(self.node)
        ne = len(self.bar)
        n_dof = self.dof * nn
        d = self.node[self.bar[:, 1], :] - self.node[self.bar[:, 0], :]
        length = np.sqrt((d ** 2).sum(axis=1))
        length = np.array([0.00001 if value == 0 else value for value in length])
        angle = d.T / length
        a = np.concatenate((-angle.T, angle.T), axis=1)
        ss = np.zeros([n_dof, n_dof])
        for k in range(ne):
            aux = self.dof * self.bar[k, :]
            index = np.r_[aux[0]:aux[0] + self.dof, aux[1]:aux[1] + self.dof]
            es = np.dot(a[k][np.newaxis].T * self.young_modulus * self.section[k], a[k][np.newaxis]) / length[k]
            ss[np.ix_(index, index)] = ss[np.ix_(index, index)] + es
        free_dof = self.support.flatten().nonzero()[0]
        kff = ss[np.ix_(free_dof, free_dof)]
        pf = self.load.flatten()[free_dof]
        uf = np.linalg.solve(kff, pf)
        u = self.support.astype(float).flatten()
        u[free_dof] = uf
        u = u.reshape(nn, self.dof)
        u_ele = np.concatenate((u[self.bar[:, 0]], u[self.bar[:, 1]]), axis=1)
        self.force = self.young_modulus * self.section / length * (a * u_ele).sum(axis=1)
        self.stress = self.force / self.section
        self.stress = np.array(self.stress)
        self.displacement = u
        self.weight = (self.density * self.section * length).sum()
        if buckling_factor is None:
            buckling_factor = 1
        self.buckling = buckling_factor * self.young_modulus * self.section[:] / length[:] ** 2
        return self.stress, self.buckling, self.displacement, self.weight

    def undeformed(self, figure, properties, input_node=None):
        # properties = [color, linestyle, linewidth, legend, legend_size]
        color = properties[0]
        linestyle = properties[1]
        linewidth = properties[2]
        if input_node is None:
            input_node = self.node
        plt.figure(figure)
        if self.truss_system == '2d':
            ax = plt.gca()
            ax.set_aspect('equal')
            for i in range(len(self.bar)):
                xi, xf = input_node[self.bar[i, 0], 0], input_node[self.bar[i, 1], 0]
                yi, yf = input_node[self.bar[i, 0], 1], input_node[self.bar[i, 1], 1]
                plt.plot([xi, xf], [yi, yf], color=color, linestyle=linestyle, linewidth=linewidth)
        else:
            plt.subplot(projection=self.truss_system)
            for i in range(len(self.bar)):
                xi, xf = input_node[self.bar[i, 0], 0], input_node[self.bar[i, 1], 0]
                yi, yf = input_node[self.bar[i, 0], 1], input_node[self.bar[i, 1], 1]
                zi, zf = input_node[self.bar[i, 0], 2], input_node[self.bar[i, 1], 2]
                plt.plot([xi, xf], [yi, yf], [zi, zf], color=color, linestyle=linestyle, linewidth=linewidth)

    def deformed(self, figure, properties, scale=None):
        if scale is None:
            scale = 1
        Truss.analysis(self)
        node = self.node + self.displacement * scale
        return self.undeformed(figure, properties, node)


# SWARM-INTELLIGENCE ALGORITHM
class Optimization:
    def __init__(self, section_variable, layout_variable, truss, population, iteration, gls_iteration):
        if len(section_variable) == 0:
            section_variable = np.ones([len(truss.bar), 2]).astype(int)
            for i in range(2):
                factor = np.linspace(0, len(truss.bar) - 1, len(truss.bar))
                section_variable[:, i] = section_variable[:, i] * factor[:]
        self.section_variable = section_variable
        self.layout_variable = layout_variable
        self.truss = truss
        self.population = population
        self.iteration = iteration
        self.gls_iteration = gls_iteration

        self.section_var_num = len(np.unique(self.section_variable[:, 0]))
        self.layout_var_num = len(np.unique(self.layout_variable[:, 0]))
        self.dimension = self.section_var_num + self.layout_var_num

        self.gbest = np.zeros(population)
        self.gbest_cost = 0
        self.gbest_iteration = np.zeros(iteration + gls_iteration)

        self.stress = np.zeros([population, len(truss.bar)])
        self.buckling = np.zeros([population, len(truss.bar)])
        self.displacement = np.zeros([population, len(truss.node), truss.dof])

    # Update section and layout variables
    @staticmethod
    def update_variable(position, section_var_num, truss, section_list, section_variable, layout_variable):
        # Update section variable
        for i in range(len(truss.bar)):
            index = int(position[section_variable[i, 0]])
            truss.section[section_variable[i, 1]] = section_list[index]
        # Update layout variable
        for i in range(len(layout_variable)):
            index = section_var_num + layout_variable[i, 0]
            truss.node[layout_variable[i, 1], layout_variable[i, 2]] = layout_variable[i, 3] * position[index]

    # Constraint control
    @staticmethod
    def constraint_check(cost, truss, stress, stress_max, buckling=None, displacement=None, displacement_max=None):
        c_total = 0
        for i in range(len(stress)):
            c_stress1 = 0
            c_stress2 = 0
            c_buckling = 0
            if stress[i] >= 0:
                if np.abs(stress[i]) > stress_max:
                    c_stress1 = np.abs((np.abs(stress[i]) - stress_max) / stress_max)
            else:
                if np.abs(stress[i]) > 15:
                    c_stress2 = np.abs((np.abs(stress[i]) - 15) / 15)
            if buckling is not None:
                if stress[i] < 0:
                    if np.abs(stress[i]) > buckling[i]:
                        c_buckling = np.abs((np.abs(stress[i]) - buckling[i]) / buckling[i])
            c_total = c_total + c_stress1 + c_stress2 + c_buckling
        if displacement is not None:
            for i in range(len(truss.nodes)):
                c_displacement = 0
                for axis in range(truss.DOF):
                    if np.abs(displacement[i, axis]) > displacement_max:
                        c_displacement = np.abs((displacement[i, axis] - displacement_max) / displacement_max)
                    c_total = c_total + c_displacement
        phi = (1 + c_total)
        cost = cost * phi
        return cost

    # Normative knowledge
    @staticmethod
    def normative_knowledge(position, dimension):
        p_upper = np.zeros(dimension)
        p_lower = np.zeros(dimension)
        for j in range(dimension):
            p_upper[j] = max(position[:, j])
            p_lower[j] = min(position[:, j])
        return p_upper, p_lower

    # Update mk
    @staticmethod
    def update_mk(p_upper, p_lower, dimension, alpha, beta, position_bound):
        mk = 0
        for j in range(dimension):
            if (p_upper[j] - p_lower[j] <= alpha * (position_bound[j, 1] - position_bound[j, 0])
                    and p_upper[j] - p_lower[j] <= beta):
                mk += 1
        return mk

    # Update learning probabilities
    @staticmethod
    def update_learning_prob(pbest_cost, mk, population, dimension):
        lmin = 0.05
        lmax = lmin + 0.25 + 0.45 * math.log(mk + 1, dimension + 1)
        pci = np.zeros(population)
        rank = np.copy(pbest_cost)
        arg_sort = np.argsort(rank)
        k = np.argsort(arg_sort) + 1
        for i in range(population):
            pci[i] = lmin + (lmax - lmin) * (math.exp(10 * (k[i] - 1) / (population - 1)) - 1) / (math.exp(10) - 1)
        return pci

    # Check if position is in search space
    @staticmethod
    def position_check(position, position_bound, dimension):
        k = 0
        for j in range(dimension):
            if position_bound[j, 0] <= position[j] <= position_bound[j, 1]:
                k += 1
            else:
                k = 0
        if k == dimension:
            return True
        else:
            return False

    # Find nearest number in list
    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        index = (np.abs(array - value)).argmin()
        return array[index]

    # Update exemplar index
    @staticmethod
    def update_exemplar(pbest_cost, pci, population, dimension, i):
        exemplar_index = np.zeros(dimension).astype(int)
        for j in range(dimension):
            if np.random.rand() < pci:
                f1 = np.random.randint(population)
                f2 = np.random.randint(population)
                while f1 == i:
                    f1 = np.random.randint(population)
                while f2 == i:
                    f2 = np.random.randint(population)
                if pbest_cost[f1] < pbest_cost[f2]:
                    exemplar_index[j] = f1
                else:
                    exemplar_index[j] = f2
            else:
                exemplar_index[j] = i
        if all(element == i for element in exemplar_index):
            ff = np.random.randint(population)
            while ff == i:
                ff = np.random.randint(population)
            exemplar_index[np.random.randint(dimension)] = ff
        return exemplar_index

    # Enhanced comprehensive learning particle swarm optimization
    def eclpso(self, section_list, layout_bound, stress_max, buckling_factor=None):
        # Parameters
        # c = 1.5
        c1 = 2
        c2 = 2
        alpha = 0.01
        beta = 2
        # w = np.linspace(0.9, 0.4, self.iteration)
        wpbe = 0.5
        g = 7

        section_int = np.linspace(0, len(section_list) - 1, len(section_list)).astype(int)

        # Position and velocity bounds
        position_bound = [min(section_int), max(section_int)] * np.ones([self.section_var_num, 2])
        position_bound = np.concatenate([position_bound, layout_bound], axis=0)
        velocity_bound = np.zeros([self.dimension, 2])
        velocity_bound[:, 0] = - (position_bound[:, 1] - position_bound[:, 0]) * 0.2
        velocity_bound[:, 1] = - velocity_bound[:, 0]

        # Initialization data
        position = np.zeros([self.population, self.dimension])
        velocity = np.zeros([self.population, self.dimension])
        cost = np.zeros(self.population)
        for i in range(self.population):
            for j in range(self.dimension):
                if j < self.section_var_num:
                    position[i, j] = random.choice(section_int[section_int > 30])
                else:
                    position[i, j] = random.uniform(position_bound[j, 0], position_bound[j, 1])
                velocity[i, j] = random.uniform(velocity_bound[j, 0], velocity_bound[j, 1])
        for i in range(self.population):
            self.update_variable(position[i], self.section_var_num, self.truss, section_list,
                                 self.section_variable, self.layout_variable)
            self.truss.analysis(buckling_factor)
            cost[i] = self.truss.weight
        pbest = np.copy(position)
        pbest_cost = np.copy(cost)
        index = np.argmin(pbest_cost)
        self.gbest = pbest[index]
        self.gbest_cost = pbest_cost[index]

        # Start iteration
        pci = np.zeros(self.population)
        stag = np.zeros(self.population)
        f = np.zeros([self.population, self.dimension]).astype(int)

        for k in range(self.iteration):
            p_upper, p_lower = self.normative_knowledge(pbest, self.dimension)
            mk = self.update_mk(p_upper, p_lower, self.dimension, alpha, beta, position_bound)
            if any(element == 0 for element in stag):
                pci = self.update_learning_prob(pbest_cost, mk, self.population, self.dimension)

            for i in range(self.population):
                if stag[i] == 0:
                    f[i] = self.update_exemplar(pbest_cost, pci[i], self.population, self.dimension, i)
                for j in range(self.dimension):
                    r1 = random.random()
                    r2 = random.random()
                    second_term = c2 * r2 * (self.gbest[j] - position[i, j])
                    if (p_upper[j] - p_lower[j] <= alpha * (position_bound[j, 1] - position_bound[j, 0])
                            and p_upper[j] - p_lower[j] <= beta):
                        n = random.normalvariate(1, 0.65)
                        perturbation_term = n * ((p_upper[j] + p_lower[j]) / 2 - pbest[f[i, j], j])
                        first_term = c1 * r1 * (pbest[f[i, j], j] + perturbation_term - position[i, j])
                        velocity[i, j] = wpbe * velocity[i, j] + first_term + second_term
                    else:
                        first_term = c1 * r1 * (pbest[f[i, j], j] - position[i, j])
                        velocity[i, j] = wpbe * velocity[i, j] + first_term + second_term
                    velocity[i, j] = min(velocity_bound[j, 1], max(velocity_bound[j, 0], velocity[i, j]))
                position[i] = position[i] + velocity[i]

                if self.position_check(position[i], position_bound, self.dimension):
                    for p in range(self.section_var_num):
                        position[i, p] = self.find_nearest(section_int, position[i, p])
                    self.update_variable(position[i], self.section_var_num, self.truss, section_list,
                                         self.section_variable, self.layout_variable)
                    self.truss.analysis(buckling_factor)
                    cost[i] = self.truss.weight
                    self.stress[i] = self.truss.stress
                    self.buckling[i] = self.truss.buckling
                    cost[i] = self.constraint_check(cost[i], self.truss, self.stress[i], stress_max, self.buckling[i])
                    if cost[i] < pbest_cost[i]:
                        pbest[i] = position[i]
                        pbest_cost[i] = cost[i]
                        stag[i] = (stag[i] + 1) % g
                        if pbest_cost[i] < self.gbest_cost:
                            self.gbest = pbest[i]
                            self.gbest_cost = pbest_cost[i]
                    else:
                        stag[i] = (stag[i] + 1) % g
            self.gbest_iteration[k] = self.gbest_cost

        # Gaussian Local Search
        index = np.argsort(pbest_cost)[0:5]
        pbest_gls = pbest[index]
        for loop in range(self.gls_iteration):
            for i in range(len(index)):
                gbest_gls = np.random.normal(self.gbest, np.abs(self.gbest - pbest_gls[i]))
                for j in range(self.dimension):
                    gbest_gls[j] = min(position_bound[j, 1], max(position_bound[j, 0], gbest_gls[j]))
                for p in range(self.section_var_num):
                    gbest_gls[p] = self.find_nearest(section_int, gbest_gls[p])
                self.update_variable(gbest_gls, self.section_var_num, self.truss, section_list,
                                     self.section_variable, self.layout_variable)
                self.truss.analysis(buckling_factor)
                gbest_gls_cost = self.truss.weight
                gbest_gls_stress = self.truss.stress
                gbest_gls_buckling = self.truss.buckling
                gbest_gls_cost = self.constraint_check(gbest_gls_cost, self.truss, gbest_gls_stress, stress_max,
                                                       gbest_gls_buckling)
                if gbest_gls_cost < self.gbest_cost:
                    self.gbest = gbest_gls
                    self.gbest_cost = gbest_gls_cost
            self.gbest_iteration[self.iteration + loop] = self.gbest_cost

    # Performance optimization
    def perform(self, run_time, section_list, layout_bound, stress_max, buckling_factor=None):
        truss_original = copy.deepcopy(self.truss)
        gbest_list = np.zeros([run_time, self.dimension])
        gbest_cost_list = np.zeros(run_time)
        gbest_cost_iteration_list = np.zeros([run_time, self.iteration + self.gls_iteration])
        run_duration = np.zeros(run_time)
        for i in range(run_time):
            start_time = time.time()
            self.eclpso(section_list, layout_bound, stress_max, buckling_factor)
            gbest_list[i] = self.gbest
            gbest_cost_list[i] = self.gbest_cost
            gbest_cost_iteration_list[i] = self.gbest_iteration
            run_duration[i] = time.time() - start_time
        gbest_cost = np.min(gbest_cost_list)
        gbest_index = np.argmin(gbest_cost_list)
        gbest = gbest_list[gbest_index]
        gbest_cost_iteration = gbest_cost_iteration_list[gbest_index]
        gbest_cost_mean = np.zeros(self.iteration + self.gls_iteration)
        run_duration_gbest = run_duration[gbest_index]
        for i in range(self.iteration + self.gls_iteration):
            gbest_cost_mean[i] = np.mean(gbest_cost_iteration_list[:, i])
        self.update_variable(gbest, self.section_var_num, self.truss,
                             section_list, self.section_variable, self.layout_variable)
        
        self.truss.analysis(buckling_factor)

        # Plot original and optimal shape of structure
        property_original = ['grey', '--', 1]
        property_optimized = ['red', '-', 1]
        truss_original.undeformed('Truss Structure', property_original)
        self.truss.undeformed('Truss Structure', property_optimized)
        structure_file = os.path.join(os.path.dirname(__file__), '1. Truss.svg')
        plt.savefig(structure_file, dpi=1200, format='svg')

        # Plot convergence curve of optimization
        plt.figure('Convergence Curve')
        plt.plot(gbest_cost_iteration, label='Best value', color='r')
        plt.plot(gbest_cost_mean, label='Mean value', color='b')
        plt.xlabel('Numbers of iteration')
        plt.ylabel('Weight (lb)')
        plt.legend()
        curve_file = os.path.join(os.path.dirname(__file__), '2. Curve.svg')
        plt.savefig(curve_file, dpi=1200, format='svg')
        np.set_printoptions(precision=4, suppress=True)
        
        Final_result = {'gbest_cost_iteration': gbest_cost_iteration, 'gbest_cost_mean':  gbest_cost_mean, 
        'Area of each element':  self.truss.section[np.newaxis].T, 'Stress Values':  self.truss.stress[np.newaxis].T, 'Displacement_X': self.truss.node[:,0], 'Displacement_Y':  self.truss.node[:,1]}
        df = pd.DataFrame.from_dict(Final_result, orient='index') 
        df1_transposed = df.T
        df1_transposed.to_excel (r'C:\Users\ADMIN\Desktop\Data_47_Bars.xlsx', index = False, header = True)

        # Print result to a text file
        print('Optimization Result')
        print('Run time = %d' % run_time)
        print('Population = %d' % self.population)
        print('Number of iteration = %d' % self.iteration)
        print('Optimal Nodal Coordinates')
        print(self.truss.node)
        print('Optimal Section')
        print(self.truss.section[np.newaxis].T)
        print('Member Stress')
        print(self.truss.stress[np.newaxis].T)
        print('Buckling')
        print(self.truss.buckling[np.newaxis].T)
        print('Best cost value = {:.4f}'.format(gbest_cost))
        print('Mean cost value = {:.4f}'.format(np.mean(gbest_cost_list)))
        print('Worst cost value = {:.4f}'.format(np.max(gbest_cost_list)))
        print('Standard deviation = {:.2f}'.format(np.std(gbest_cost_list)))
        print('Run duration = {:.2f} seconds'.format(run_duration_gbest))
        
        return gbest_cost_list

# Input of application
modulus_elasticity = 3e4
material_density = 0.3
type_of_structure = '2d'

nodes = np.array([[-60, 0],
                  [60, 0],
                  [-60, 120],
                  [60, 120],
                  [-60, 240],
                  [60, 240],
                  [-60, 360],
                  [60, 360],
                  [-30, 420],
                  [30, 420],
                  [-30, 480],
                  [30, 480],
                  [-30, 540],
                  [30, 540],
                  [-90, 570],
                  [90, 570],
                  [-150, 600],
                  [-90, 600],
                  [-30, 600],
                  [30, 600],
                  [90, 600],
                  [150, 600]])

bars = np.array([[8, 10],
                 [10, 12],
                 [9, 11],
                 [11, 13],
                 [8, 11],
                 [9, 10],
                 [10, 11],
                 [10, 13],
                 [11, 12],
                 [12, 13],
                 [12, 14],
                 [13, 15],
                 [12, 18],
                 [13, 19],
                 [13, 18],
                 [12, 19],
                 [14, 16],
                 [15, 21],
                 [14, 18],
                 [15, 19],
                 [14, 17],
                 [15, 20],
                 [16, 17],
                 [20, 21],
                 [17, 18],
                 [19, 20],
                 [18, 19],
                 [8, 9],
                 [6, 8],
                 [7, 9],
                 [6, 9],
                 [7, 8],
                 [6, 7],
                 [4, 6],
                 [5, 7],
                 [4, 7],
                 [5, 6],
                 [4, 5],
                 [2, 4],
                 [3, 5],
                 [2, 5],
                 [3, 4],
                 [2, 3],
                 [0, 2],
                 [1, 3],
                 [0, 3],
                 [1, 2]])

truss_1 = Truss(modulus_elasticity, material_density, type_of_structure, nodes, bars)

loads = truss_1.load
loads[16, 0] = 6
loads[16, 1] = -14
loads[21, 0] = 6
loads[21, 1] = -14

supports = truss_1.support
supports[0, :] = 0
supports[1, :] = 0

# Truss optimization input
section = np.linspace(0.1, 5)

layout = np.array([[30, 150],
                   [30, 150],
                   [60, 180],
                   [30, 150],
                   [150, 300],
                   [30, 150],
                   [300, 390],
                   [15, 60],
                   [390, 450],
                   [15, 60],
                   [450, 510],
                   [15, 60],
                   [510, 555],
                   [0, 60],
                   [570, 660],
                   [60, 120],
                   [570, 660]])

# section variable with input [[variable number in order 0,1,2,... , bar number],[...]]
section_var = np.array([[0, 0],
                        [0, 2],
                        [1, 1],
                        [1, 3],
                        [2, 4],
                        [2, 5],
                        [3, 6],
                        [4, 7],
                        [4, 8],
                        [5, 9],
                        [6, 10],
                        [6, 11],
                        [7, 12],
                        [7, 13],
                        [8, 14],
                        [8, 15],
                        [9, 16],
                        [9, 17],
                        [10, 18],
                        [10, 19],
                        [11, 20],
                        [11, 21],
                        [12, 22],
                        [12, 23],
                        [13, 24],
                        [13, 25],
                        [14, 26],
                        [15, 27],
                        [16, 28],
                        [16, 29],
                        [17, 30],
                        [17, 31],
                        [18, 32],
                        [19, 33],
                        [19, 34],
                        [20, 35],
                        [20, 36],
                        [21, 37],
                        [22, 38],
                        [22, 39],
                        [23, 40],
                        [23, 41],
                        [24, 42],
                        [25, 43],
                        [25, 44],
                        [26, 45],
                        [26, 46]])

# layout variable with input [[variable number in order 0,1,2,... , node number, axis X or Y],[...]]
layout_var = np.array([[0, 0, 0, -1],
                       [0, 1, 0, 1],
                       [1, 2, 0, -1],
                       [1, 3, 0, 1],
                       [2, 2, 1, 1],
                       [2, 3, 1, 1],
                       [3, 4, 0, -1],
                       [3, 5, 0, 1],
                       [4, 4, 1, 1],
                       [4, 5, 1, 1],
                       [5, 6, 0, -1],
                       [5, 7, 0, 1],
                       [6, 6, 1, 1],
                       [6, 7, 1, 1],
                       [7, 8, 0, -1],
                       [7, 9, 0, 1],
                       [8, 8, 1, 1],
                       [8, 9, 1, 1],
                       [9, 10, 0, -1],
                       [9, 11, 0, 1],
                       [10, 10, 1, 1],
                       [10, 11, 1, 1],
                       [11, 12, 0, -1],
                       [11, 13, 0, 1],
                       [12, 12, 1, 1],
                       [12, 13, 1, 1],
                       [13, 18, 0, -1],
                       [13, 19, 0, 1],
                       [14, 18, 1, 1],
                       [14, 19, 1, 1],
                       [15, 17, 0, -1],
                       [15, 20, 0, 1],
                       [16, 17, 1, 1],
                       [16, 20, 1, 1]])

ps = 20
iter_max = 1500
gls_iter = 1
stress_limit = 20.00
k_eff = 3.96
run = 15

algorithm = Optimization(section_var, layout_var, truss_1, ps, iter_max, gls_iter)
t = algorithm.perform(run, section, layout, stress_limit, k_eff)
print(t)
