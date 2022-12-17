#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
This file creates the graph object.
"""
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import ast
import pandas as pd
from scipy.sparse import diags
from scipy.interpolate import interp1d
import scipy.linalg as la
import scipy.optimize as opt


# In[2]:


class Graph(object):
    """
    This is the graph object. The input should be a dictionary 
    whose keys are the nodes and values are nodes connected to them via an edge.
    
    This should only be a base object, so only basics should be implemented in this class. 
    For more specific and complicated functions, implement in subclasses.
    
    Inputs
    -----------------
    graph_dict: The dictionary representation of the graph
        type: dictionary
        
    Public methods
    ------------------------
    show_gd  :  returns the graph-representing dictionary that is the input.
    
    find_vertices:  returns a set of vertices.
    
    find_vertices_list:  returns a list of vertices.
    
    find_edges： returns a list of edges represented in set of starting point and ending point. 
    
    find_edges_list： returns a list of edges represented as lists.
    
    edges_of_vertex： returns a list of all the edges of a vertice.

    add_vertex： adds a vertex.
    
    add_edge: adds an edge.
    
    visualize: visualize the graph.
    
    """
    
    def __init__(self, graph_dict = None):
        """
        The input of this graph object should be a dictionary, 
        which by default is empty to prevent errors.
        As a local convension, I use the line 
        
        gd = self.graph_dict 
        
        in each of the implemented method.
        """
        if graph_dict == None:
            graph_dict = {}
        self.graph_dict = graph_dict
        
    def show_gd(self):
        gd = self.graph_dict
        return gd
        
    def _find_vertices(self):
        gd = self.graph_dict
        return set(gd.keys())
        
    def _find_edges(self):
        """ 
        A static method generating the edges of the graph "graph". 
        Edges are represented as sets with one (a loop back to the vertex) or two vertices.
        A little test was made to avoid repetition.
        
        For ordered edges, it's better to write in lists. 
        But that will be done in subclasses.
        """
        gd = self.graph_dict
        edges = []
        for start_pt in gd:
            for end_pt in gd[start_pt]:
                if [end_pt, start_pt] not in edges:
                    edges.append([start_pt, end_pt])
        return edges
        
    def find_vertices(self):
        """
        It is convenient to use public method to initiate find vertices process, 
        but use private method to really find them.
        """
        return self._find_vertices()
    
    def find_vertices_list(self):
        """
        This returns a list of vertices.
        """
        return list(self._find_vertices())
    
    def find_edges(self):
        return self._find_edges()
    
    def edges(self, vertice):
        """ 
        returns a list of all the edges of a vertice.
        """
        gd = self.graph_dict
        return gd[vertice]
    
    
    def add_vertex(self, vertex):
        """ 
        We can add a vertex with this method. 
        Note that the added value of the key(vertex) is a set.
        """
        gd = self.graph_dict
        if vertex not in gd:
            gd[vertex] = set()
            
    def add_edge(self, start_pt, end_pt):
        """ 
        Input the start point and end point. 
        Then we add this edge if its not already there.
        Note that we can add a line with nodes not already in the graph.
        
        For multiple edges between two nodes, implement another method/another class.
        """
        gd = self.graph_dict
        if end_pt not in gd:
            gd[end_pt] = set()
        if start_pt in gd:
            gd[start_pt].add(end_pt)
            gd[end_pt].add(start_pt)
        else:
            gd[start_pt] = set(end_pt)
            gd[end_pt].add(start_pt)
    def visualize(self):
        "This visualizes the graph"
        return nx.draw(nx.Graph(self.show_gd()), with_labels = True)
    
    


# In[3]:


class Metric_Graph(Graph):
    """
    This is the metric graph object. It's a subclass of the Graph object. 
    In addition to the graph dictionary, there's also an edge-interval dictionary input here.
    
    This should only be another base object. So we'll not solve equations directly here.
    
    Inputs
    -----------------
    graph_dict: The dictionary representation of the graph
        type: dictionary
        
    edge_interval_dictionary: The dictionary representation of the intervals on each edge
        example: {"['a', 'c']": [0, 1]}
        type: dictionary
        
    Public methods
    ------------------------
    show_interval  :  returns the edge-interval dictionary that is the additional input.
    
    show_vertex_value:  This method returns the value of a vertex on the edge that it is on. 
                            If edge not specified, then return all edge it is on, and it's value on it.
        input syntax: .show_vertex_value('f',['f', 'g'])
    
    The rest are inherent from Graph object.
    
    """
    
    def __init__(self, graph_dict = None, edge_interval_dict = None):
        super(Metric_Graph, self).__init__(graph_dict)
        if edge_interval_dict == None:
            edge_interval_dict = {}
        self.edge_interval_dict = edge_interval_dict
    
    def show_interval(self):
        eid = self.edge_interval_dict
        return eid
    
    def show_vertex_value(self, vertex, edge = None):
        """
        This method reports the value of a vertex on any edge it is on. You can also pick the specific edge. 
        The syntax is this: .show_vertex_value('f',['f', 'g'])
        This method uses pandas since it's the way to do data processing, we think.
        """
        interval1 = self.edge_interval_dict
        st = []
        stv = []
        edv = []
        for i in interval1:
            st.append(eval(i))
            stv.append(interval1[i][0])
            edv.append(interval1[i][1])
        df = pd.DataFrame(st, columns = ["start","end"])
        df["start value"] = stv
        df["ending value"] = edv
        x = vertex
        astr = x
        df_newf_1 = df[df["start"] == astr]
        df_newf_2 = df[df["end"] == astr]
        dff = pd.merge(df_newf_1, df_newf_2,how = "outer")
        if edge == None:
            for i in range(dff.shape[0]):
                if dff.iloc[i,0] == x:
                    print("The value of "+x+" is {}".format(dff.iloc[i,2])+" in the edge [{},{}]".format(dff.iloc[i,0],dff.iloc[i,1]))
                if dff.iloc[i,1] == x:
                    print("The value of "+x+" is {}".format(dff.iloc[i,3])+" in the edge [{},{}]".format(dff.iloc[i,0],dff.iloc[i,1]))
            return
        else:
            if x == edge[0]:
                for i in range(dff.shape[0]):
                    if dff.iloc[i,0] == x and dff.iloc[i,1] == edge[1]:
                        print("The value of "+x+" is {}".format(dff.iloc[i,2])+" in the edge [{},{}]".format(dff.iloc[i,0],dff.iloc[i,1]))
                        return
            else:
                for i in range(dff.shape[0]):
                    if dff.iloc[i,1] == x and dff.iloc[i,0] == edge[1]:
                        print("The value of "+x+" is {}".format(dff.iloc[i,3])+" in the edge [{},{}]".format(dff.iloc[i,0],dff.iloc[i,1]))
                        return


# In[ ]:





# In[4]:


def sign_helper(index, sub_df): 
    #at this index, if the node is a starting node, then the sign of f' is + in the sum
    if sub_df.at[index, 'ending value'] > sub_df.at[index, 'starting value']:
        return -1
    else:
        return 1
        


# In[5]:


def node_start_subdf(nodename, df):
    sub_df = df[df["start"] == nodename]
    return sub_df


# In[6]:


def node_end_subdf(nodename, df):
    sub_df = df[df["end"] == nodename]
    return sub_df


# In[7]:


### u'' = f case

def solve_u(vertices,df, nx = 41):  
    '''
    for convenience, vertices must be a list, where orders don't matter.
    '''
    p = len(vertices)
    m = df.shape[0]
    
    M = np.zeros((m*nx, m*nx))
    diagonals = [[1.], [-2.], [1.]]
    offsets = [0, 1, 2]
    d2mat = diags(diagonals, offsets, shape=(nx,nx)).toarray()
    middle = d2mat[0:-2, :]
    
    
    b = np.zeros(m*nx)
    ### We have m * (nx - 2) linear equations for second order derivative conditions
    
    for k in range(m):
        ### nx is fixed, and each intervals (i.e., edges) have their own dx
        
        intk_s = df.at[k, 'starting value']
        intk_e = df.at[k, 'ending value']
        dx = abs(intk_e -intk_s)/(nx-1)
        M[k*(nx-2):(k+1)*(nx-2), k*nx:(k+1)*nx] = middle/dx**2
        
        fk = df.at[k, 'function']
        fvals = fk(np.linspace(intk_s, intk_e, nx))[1:-1]
        b[k*(nx-2):(k+1)*(nx-2)] = fvals
    
    
    start_ind = 0
    
    for i in range(p):
        node_name = vertices[i]
        
        start_subdf = node_start_subdf(node_name,df)
        end_subdf = node_end_subdf(node_name,df)
        
        col_start_index = []
        col_end_index = []
    ### we construct p linear equations for f' condition (stacked at the bottom of the matrix)
        for s in start_subdf.index:
            int_len = abs(start_subdf.at[s,'ending value'] - start_subdf.at[s,'starting value'])
            dx = int_len/(nx-1)
            
            sid = nx*s
            col_start_index.append(sid)
            ### forward difference for f'
            M[nx*m - p + i, sid:(sid+3)] = np.array([-1.5, 2, -0.5])/dx * sign_helper(s, start_subdf)
            
        for e in end_subdf.index:
            int_len = abs(end_subdf.at[e,'ending value'] - end_subdf.at[e,'starting value'])
            dx = int_len/(nx-1)
            
            eid = (e+1)*(nx) -1 
            col_end_index.append(eid)
            
            ### backward difference for f'
            M[nx*m - p + i, (eid-2):(eid+1)] = np.array([0.5, -2, 1.5])/dx * -sign_helper(e, end_subdf)
        
    ### Next, construct 2*m - p linear equations for boundary conditions (values match at each vertices)
        ### Recall: sum of degrees d_i = 2m. So summation of d_i - 1  over i = 1,...,p verticies gives 2*m - p
        col_index = col_start_index + col_end_index
        deg = start_subdf.shape[0] + end_subdf.shape[0]
        
        M[m*(nx-2) + start_ind: m*(nx-2) + start_ind + deg - 1, col_index[0]] = 1
        
        for d in range(deg - 1):
            M[m*(nx-2) + start_ind + d, col_index[d+1]] = - 1
        
        start_ind += deg - 1
    
            
    M_inv = np.linalg.pinv(M)
    return M, M_inv@b


# In[8]:


### u'' - u = f case

def solve_u_lambda( vertices, df, nx = 41):
    '''
    for convenience, vertices must be a list, where orders don't matter.
    '''
    p = len(vertices)
    m = df.shape[0]
    
    M = np.zeros((m*nx, m*nx))
    diagonals = [[1.], [-2.], [1.]] 
    offsets = [0, 1, 2]
    d2mat = diags(diagonals, offsets, shape=(nx,nx)).toarray()
    middle = d2mat[0:-2, :]
    
    u_idmat = diags([[1]], [1], shape = (nx, nx)).toarray()[0:-2, :] #similarly, u corresponds to a matrix with 0 and 1
    
    
    b = np.zeros(m*nx)
    ### We have m * (nx - 2) linear equations for second order derivative conditions
    
    for k in range(m):
        ### nx is fixed, and each intervals (i.e., edges) have their own dx
        
        intk_s = df.at[k, 'starting value']
        intk_e = df.at[k, 'ending value']
        dx = abs(intk_e -intk_s)/(nx-1)
        M[k*(nx-2):(k+1)*(nx-2), k*nx:(k+1)*nx] = middle/dx**2 - u_idmat   # u'' - u
        
        fk = df.at[k, 'function']
        fvals = fk(np.linspace(intk_s, intk_e, nx))[1:-1]
        b[k*(nx-2):(k+1)*(nx-2)] = fvals
    
    
    start_ind = 0
    
    for i in range(p):
        node_name = vertices[i]
        
        start_subdf = node_start_subdf(node_name,df)
        end_subdf = node_end_subdf(node_name,df)
        
        col_start_index = []
        col_end_index = []
    ### we construct p linear equations for f' condition (stacked at the bottom of the matrix)
        for s in start_subdf.index:
            int_len = abs(start_subdf.at[s,'ending value'] - start_subdf.at[s,'starting value'])
            dx = int_len/(nx-1)
            
            sid = nx*s
            col_start_index.append(sid)
            ### forward difference for f'
            M[nx*m - p + i, sid:(sid+3)] = np.array([-1.5, 2, -0.5])/dx * sign_helper(s, start_subdf)
            
        for e in end_subdf.index:
            int_len = abs(end_subdf.at[e,'ending value'] - end_subdf.at[e,'starting value'])
            dx = int_len/(nx-1)
            
            eid = (e+1)*(nx) -1 
            col_end_index.append(eid)
            
            ### backward difference for f'
            M[nx*m - p + i, (eid-2):(eid+1)] = np.array([0.5, -2, 1.5])/dx * -sign_helper(e, end_subdf)
        
    ### Next, construct 2*m - p linear equations for boundary conditions (values match at each vertices)
        ### Recall: sum of degrees d_i = 2m. So summation of d_i - 1  over i = 1,...,p verticies gives 2*m - p
        col_index = col_start_index + col_end_index
        deg = start_subdf.shape[0] + end_subdf.shape[0]
        
        M[m*(nx-2) + start_ind: m*(nx-2) + start_ind + deg - 1, col_index[0]] = 1
        
        for d in range(deg - 1):
            M[m*(nx-2) + start_ind + d, col_index[d+1]] = - 1
        
        start_ind += deg - 1
    
            
    M_inv = np.linalg.pinv(M)
    return M, M_inv@b   
    


# In[9]:


def find_edge_sol(T, df, edge, nx = 41):
    '''
    T: solution array from the linear system
    edge: 2 element list of nodes names; ordered
    nx: number of points of each edge used for solution T
    '''
    idx = df[(df["start"] == edge[0]) & (df["end"] == edge[1])].index.tolist()[0]
    xs = np.linspace(df.at[idx, 'starting value'], df.at[idx, 'ending value'], nx)
    return xs, T[idx*nx:(idx+1)*nx]


# In[ ]:





# In[18]:


class Graph_PDE(Metric_Graph):
    """
    This is the quantum graph object. It's a subclass of the Metric_Graph object. 
    In addition to the graph dictionary and the edge-interval dictionary, there's also an edge-pde input here.
    (Even though we always input ODE here, there really is potential to solve PDEs and that's our initial goal, so we put it here.)
    
    We will implement two methods of solving the system here. 
    The first is self-contained solver of the system using discretization.
    The other changes the problem into a bvp problem plus an optimization problem.
    
    Inputs
    -----------------
    graph_dict: The dictionary representation of the graph
        type: dictionary
        
    edge_interval_dictionary: The dictionary representation of the intervals on each edge
        example: {"['a', 'c']": [0, 1]}
        type: dictionary
        
    edge_pde_dictionary: The dictionary representation of the right-hand-side of the pde on each edge
        example: {   "['a', 'b']" : f}
        type: dictionary
        
    Public methods
    ------------------------------------------------------------------------------
    find_edges_direction: returns the needed lists for the dataframe, users can omit
    
    panda: returns the dataframe needed for finite difference
    
    
    for the equation Laplacian(u) = f, using finite difference
    -------------------------------------------------------------
    
    See_matrix: returns the huge matrix used for finite differnce method.
    
    solve_finite: returns the concatenated solution of the graph, the order is not as the one in graph so use below methods to see solution.
    
    solve_finite_edge: returns the solution of a particular edge.
    
    solve_finite_plot: plots the solution of the system on the particular edge.
    
    
    Below is repeating above step for the equation Laplacian(u) + u = f, using finite difference
    -------------------------------------------------------------------------------------------------
    See_matrix_lambda: returns the huge matrix used for finite differnce method.
    
    solve_finite_lambda: returns the concatenated solution of the graph, the order is not as the one in graph so use below methods to see solution.
    
    solve_finite_edge_lambda: returns the solution of a particular edge.
    
    solve_finite_plot_lambda: plots the solution of the system on the particular edge.
    
    
    for the equation Laplacian(u) = f, using optimization
    -------------------------------------------------------------
    minimize_sol  :  returns the optimization solution for the equation Laplacian u = f, 
                        as well as a BVP dictionary that can be used to implement the whole object as 
                        a Graph_PDE_BVP object. This is mostly just for other methods, but one can access it too.
                        
                    Input is the initial condition. We've discovered that this method is really sensitive denpendent, 
                        so it's best to enter an initial guess rather than use the default [0,0,0].
    
    minimize_solution_BVP_obj:  This method returns the Graph_PDE_BVP object 
                                    generated with the BVP-dictionary using the above minimize_sol.
                                
    minimize_solution_edge: This method returns the solution on a specific edge.
            input syntax: .minimize_solution_edge(["a","b"], [0,0,0])
            
    minimize_solution_plot: This method plots a discrete plot of the solution on some edge, with some initial guess.
    
    minimize_solution_plot_curve: This method plots a continuous plot of the solution on some edge, with some initial guess.
    
    
    Below is repeating above step for the equation Laplacian(u) + u = f, using optimization
    -------------------------------------------------------------------------------------------------
    
    
    minimize_sol_lambda  :  returns the optimization solution for the equation Laplacian(u) + u = f, 
                        as well as a BVP dictionary that can be used to implement the whole object as 
                        a Graph_PDE_BVP object. This is mostly just for other methods, but one can access it too.
                        
                    Input is the initial condition. We've discovered that this method is really sensitive denpendent, 
                        so it's best to enter an initial guess rather than use the default [0,0,0].
    
    minimize_solution_BVP_obj_lambda:  This method returns the Graph_PDE_BVP object 
                                    generated with the BVP-dictionary using the above minimize_sol.
                                
    minimize_solution_edge_lambda: This method returns the solution on a specific edge.
            input syntax: .minimize_solution_edge(["a","b"], [0,0,0])
            
    minimize_solution_plot_lambda: This method plots a discrete plot of the solution on some edge, with some initial guess.
    
    minimize_solution_plot_curve_lambda: This method plots a continuous plot of the solution on some edge, with some initial guess.

    
    
    The rest are inherent from Metric_Graph object.
    
    """
    
    def __init__(self, graph_dict = None, edge_interval_dict = None, PDE_edge_dict = None):
        super(Graph_PDE, self).__init__(graph_dict, edge_interval_dict)
        if PDE_edge_dict == None:
            PDE_edge_dict = {}
        self.PDE_edge_dict = PDE_edge_dict
        
    def find_edges_direction(self):
        eid = self.edge_interval_dict
        edl = [] # list of ["a","b"]
        fl = [] 
        itvl = [] 
        for i in eid:
            fl.append(self.PDE_edge_dict[i])
            edl.append(eval(i))
            itvl.append(eid[i])
        return edl, fl, itvl

            
    def panda(self):
        edl,fl, itvl = self.find_edges_direction()
        df1 = pd.DataFrame(edl, columns = ["start","end"])
        df2 = pd.DataFrame(itvl, columns = ["starting value","ending value"])
        df = pd.concat([df1,df2],axis = 1)
        df["function"] = fl
        return df
    
    def See_matrix(self):
        """
        Returns the solver matrix M.
        """
        df = self.panda()
        vertices = self.find_vertices_list()
        M, T = solve_u(vertices,df, nx = 41)
        return M
    
    def solve_finite(self):
        """
        Returns the long T solution.
        """
        df = self.panda()
        vertices = self.find_vertices_list()
        M, T = solve_u(vertices,df, nx = 41)
        return T
    
    def solve_finite_edge(self,edge):
        """
        returns the solution of a particular edge.
        """
        df = self.panda()
        vertices = self.find_vertices_list()
        M, T = solve_u(vertices,df, nx = 41)
        x, sol = find_edge_sol(T, df, edge, nx = 41)
        return sol
    
    def solve_finite_plot(self,edge):
        
        df = self.panda()
        vertices = self.find_vertices_list()
        M, T = solve_u(vertices,df, nx = 41)
        x, sol = find_edge_sol(T, df, edge, nx = 41)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(x, sol, "m", label='Approximate solution')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$T$')
        ax.set_title('Solution for equation u\'\' = f on the edge{}'.format(edge))
        ax.legend();
        return
    
        # shifted equation below
#---------------------------------------------------------------------------------------------

    def See_matrix_lambda(self):
        """
        Returns the solver matrix M.
        """
        df = self.panda()
        vertices = self.find_vertices_list()
        M, T = solve_u_lambda(vertices,df, nx = 41)
        return M
    
    def solve_finite_lambda(self):
        """
        Returns the long T solution.
        """
        df = self.panda()
        vertices = self.find_vertices_list()
        M, T = solve_u_lambda(vertices,df, nx = 41)
        return T
    
    def solve_finite_edge_lambda(self,edge):
        """
        returns the solution of a particular edge.
        """
        df = self.panda()
        vertices = self.find_vertices_list()
        M, T = solve_u_lambda(vertices,df, nx = 41)
        x, sol = find_edge_sol(T, df, edge, nx = 41)
        return sol
    
    def solve_finite_plot_lambda(self,edge):
        
        df = self.panda()
        vertices = self.find_vertices_list()
        M, T = solve_u_lambda(vertices,df, nx = 41)
        x, sol = find_edge_sol(T, df, edge, nx = 41)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(x, sol,"m", label='Approximate solution')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$T$')
        ax.set_title('Solution for equation u\'\' - u = f on the edge{}'.format(edge))
        ax.legend();
        return
        
        
        # Optimization method below
#---------------------------------------------------------------------------------------------
        
    def _net_flow_count(self, BVP_dict):
        """
        returns the net flow.
        """
        netflow = net_flow(self.graph_dict, self.edge_interval_dict, self.PDE_edge_dict, BVP_dict)
        return netflow
    
    def minimize_sol(self, x0 = None):
        """
        returns the minimize sol object of the minimization of the flow count. 
        Also returns the BVP_dict that keeps track of the vertex index.
        """
        if x0 is None:
            x0 = np.zeros(len(self.find_vertices()))
        fv = self.find_vertices_list()
        def f(x):
            BVP_dict = {}
            ct = 0
            for key in fv:
                BVP_dict[key] = x[ct]
                ct = ct +1
            return self._net_flow_count( BVP_dict)
        BVP_dict = {}
        ct = 0
        for key in fv:
            BVP_dict[key] = x0[ct]
            ct = ct +1
        sol = opt.minimize(f, x0, tol=1e-5)
        return sol, BVP_dict
    
    def minimize_solution_BVP_obj(self, x0 = None):
        """
        Returns the best minimized BVP object.
        """
        if x0 is None:
            x0 = np.zeros(len(self.find_vertices()))
        sol, BVP_dict = self.minimize_sol(x0)
        ct = 0
        for key in BVP_dict:
            BVP_dict[key] = sol.x[ct]
            ct = ct +1
        AA = Graph_PDE_BVP(self.graph_dict, self.edge_interval_dict, self.PDE_edge_dict, BVP_dict)
        return AA
    
    def minimize_solution_edge(self, edge, x0 = None):
        """
        returns the solve on edge BVP with above defined object.
        """
        AA = self.minimize_solution_BVP_obj(x0)
        return AA.solve(edge)
    
    def minimize_solution_plot(self, edge , x0 = None):
        """
        returns the plot on edge BVP with above defined object.
        """
        AA = self.minimize_solution_BVP_obj(x0)
        return AA.plot(edge)
    
    def minimize_solution_plot_curve(self, edge , x0 = None):
        """
        returns the plot on edge BVP with above defined object.
        """
        AA = self.minimize_solution_BVP_obj(x0)
        return AA.plot_curve(edge)
    
    
#    def solve_PDE_on_edge(self):
#        eid = self.edge_interval_dict
#        return eid
    
#    def solve_PDE_with_BV

    def _net_flow_count_lambda(self, BVP_dict):
        """
        returns the net flow.
        """
        netflow = net_flow_lambda(self.graph_dict, self.edge_interval_dict, self.PDE_edge_dict, BVP_dict)
        return netflow
    
    def minimize_sol_lambda(self, x0 = None):
        """
        returns the minimize sol object of the minimization of the flow count. 
        Also returns the BVP_dict that keeps track of the vertex index.
        """
        if x0 is None:
            x0 = np.zeros(len(self.find_vertices()))
        fv = self.find_vertices_list()
        def f(x):
            BVP_dict = {}
            ct = 0
            for key in fv:
                BVP_dict[key] = x[ct]
                ct = ct +1
            return self._net_flow_count_lambda( BVP_dict)
        BVP_dict = {}
        ct = 0
        for key in fv:
            BVP_dict[key] = x0[ct]
            ct = ct +1
        sol = opt.minimize(f, x0, tol=1e-5)
        return sol, BVP_dict
    
    def minimize_solution_BVP_obj_lambda(self, x0 = None):
        """
        Returns the best minimized BVP object.
        """
        if x0 is None:
            x0 = np.zeros(len(self.find_vertices()))
        sol, BVP_dict = self.minimize_sol_lambda(x0)
        ct = 0
        for key in BVP_dict:
            BVP_dict[key] = sol.x[ct]
            ct = ct +1
        AA = Graph_PDE_BVP(self.graph_dict, self.edge_interval_dict, self.PDE_edge_dict, BVP_dict)
        return AA
    
    def minimize_solution_edge_lambda(self, edge, x0 = None):
        """
        returns the solve on edge BVP with above defined object.
        """
        AA = self.minimize_solution_BVP_obj_lambda(x0)
        return AA.solve_lambda(edge)
    
    def minimize_solution_plot_lambda(self, edge , x0 = None):
        """
        returns the plot on edge BVP with above defined object.
        """
        AA = self.minimize_solution_BVP_obj_lambda(x0)
        return AA.plot_lambda(edge)
    
    def minimize_solution_plot_curve_lambda(self, edge , x0 = None):
        """
        returns the plot on edge BVP with above defined object.
        """
        AA = self.minimize_solution_BVP_obj_lambda(x0)
        return AA.plot_curve_lambda(edge)


# In[ ]:





# In[11]:


def net_flow(graph_dict, edge_interval_dict, PDE_edge_dict, BVP_dict):
    """
    returns the net flow of a known BVP graph. Laplacian(u) = f
    """
    AA = Graph_PDE_BVP(graph_dict, edge_interval_dict, PDE_edge_dict, BVP_dict)
    df = AA.panda()
    vertex_list = AA.find_vertices()
    #     The thing to do here is to use x and T, the exact solutions of our method since it might happen 
    # that TS is not a good approximation at the endpoints.
    flow = np.zeros(len(vertex_list))
    count = 0
    for a in vertex_list:
        astr = "{}".format(a)
        df_newf_1 = df[df["start"] == astr]
        df_newf_2 = df[df["end"] == astr]
        dff = pd.merge(df_newf_1, df_newf_2,how = "outer")
        dff
        # !!!!!!!! This is assuming that there's no starting and ending at same point.
        for i in range(df_newf_1.shape[0]):
            TS, x, T = AA.solve([df_newf_1.iloc[i,0],df_newf_1.iloc[i,1]])
            steplen = (df_newf_1.iloc[i,3]-df_newf_1.iloc[i,2])/(40)
            flow[count] = flow[count]+ ((-3./2.*T[0]+2.*T[1]-1./2.*T[2])/(steplen))
        for i in range(df_newf_2.shape[0]):
            TS, x, T = AA.solve([df_newf_2.iloc[i,0],df_newf_2.iloc[i,1]])
            steplen = (df_newf_2.iloc[i,3]-df_newf_2.iloc[i,2])/(40)
            flow[count] = flow[count]+ ((-3./2.*T[-1]+2.*T[-2]-1./2.*T[-3])/(steplen))
        flow[count] = abs(flow[count])
        count = count+1
        del df_newf_1
        del df_newf_2
    netflow = sum(flow)
    return netflow


# In[12]:


def net_flow_lambda(graph_dict, edge_interval_dict, PDE_edge_dict, BVP_dict):
    """
    returns the net flow of a known BVP graph. Laplacian(u) + u = f
    """
    AA = Graph_PDE_BVP(graph_dict, edge_interval_dict, PDE_edge_dict, BVP_dict)
    df = AA.panda()
    vertex_list = AA.find_vertices()
    #     The thing to do here is to use x and T, the exact solutions of our method since it might happen 
    # that TS is not a good approximation at the endpoints.
    flow = np.zeros(len(vertex_list))
    count = 0
    for a in vertex_list:
        astr = "{}".format(a)
        df_newf_1 = df[df["start"] == astr]
        df_newf_2 = df[df["end"] == astr]
        dff = pd.merge(df_newf_1, df_newf_2,how = "outer")
        dff
        # !!!!!!!! This is assuming that there's no starting and ending at same point.
        for i in range(df_newf_1.shape[0]):
            TS, x, T = AA.solve_lambda([df_newf_1.iloc[i,0],df_newf_1.iloc[i,1]])
            steplen = (df_newf_1.iloc[i,3]-df_newf_1.iloc[i,2])/(40)
            flow[count] = flow[count]+ ((-3./2.*T[0]+2.*T[1]-1./2.*T[2])/(steplen))
        for i in range(df_newf_2.shape[0]):
            TS, x, T = AA.solve_lambda([df_newf_2.iloc[i,0],df_newf_2.iloc[i,1]])
            steplen = (df_newf_2.iloc[i,3]-df_newf_2.iloc[i,2])/(40)
            flow[count] = flow[count]+ ((-3./2.*T[-1]+2.*T[-2]-1./2.*T[-3])/(steplen))
        flow[count] = abs(flow[count])
        count = count+1
        del df_newf_1
        del df_newf_2
    netflow = sum(flow)
    return netflow


# In[13]:


class Graph_PDE_BVP(Graph_PDE):
    """
    This is the bvp quantum graph object. It's a subclass of the Graph_PDE object. 
    In addition to the graph dictionary and the edge-interval dictionary, 
        edge-pde-dictionary input, there's also a vertex-value dictionary input.
    (Even though we always input ODE here, there really is potential to solve PDEs and that's our initial goal, so we put it here.)
    
    This is just solving the BVP problem, not directly solving the quantum graph problem. 
    However this is everything we need to solve the quantum graph using optimization method.
    
    Inputs
    -----------------
    graph_dict: The dictionary representation of the graph
        type: dictionary
        
    edge_interval_dictionary: The dictionary representation of the intervals on each edge
        example: {"['a', 'c']": [0, 1]}
        type: dictionary
        
    edge_pde_dictionary: The dictionary representation of the right-hand-side of the pde on each edge
        example: {   "['a', 'b']" : f}
        type: dictionary
        
    BVP_dictionary: The dictionary representation the value of each vertex of the graph. 
        example: {   "a" : 1}
        type: dictionary
        
    Public methods
    ------------------------------------------------------------------------------
    show_value: This method returns the input vertex-value dictionary.
    
    find_edges_direction: returns 3 things:
        edl: This is the list of the edges
        fl: list of functions
        itvl: list of intervals

            It's important to have these in the same method to match terms. The name is a little bit misleading,
                so be cautious.
    panda: This method returns everything as a pd.DataFrame. User can use it to check, and all following codes are based on it.
    
    
    for the equation Laplacian(u) = f
    -------------------------------------------------------------
    solve: Directly solves the bvp problem Laplacian(u) = f, using discretization method
        reference: https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/03_FiniteDifferences/03_03_BoundaryValueProblems.html
        
    plot: plot the discrete version of the solution
    
    plot_curve: plot the curve of the solution
    
    
    
    for the equation Laplacian(u) - u = f
    -------------------------------------------------------------
    solve_lambda: Directly solves the bvp problem Laplacian(u) + u = f, using discretization method
        reference: https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/03_FiniteDifferences/03_03_BoundaryValueProblems.html
        The only difference is to add an identity to the matrix
        
    plot_lambda: plot the discrete version of the solution
    
    plot_curve_lambda: plot the curve of the solution
    """
    
    def __init__(self, graph_dict = None, edge_interval_dict = None, PDE_edge_dict = None, BVP_dict = None):
        super(Graph_PDE_BVP, self).__init__(graph_dict, edge_interval_dict, PDE_edge_dict)
        if BVP_dict == None:
            BVP_dict = {}
        self.BVP_dict = BVP_dict
    
    def show_value(self):
        eidb = self.BVP_dict
        return eidb
    
    
    def find_edges_direction(self):
        eid = self.edge_interval_dict
        edl = []
        fl = []
        itvl = []
        for i in eid:
            edl.append(eval(i))
            fl.append(self.PDE_edge_dict[i])
            itvl.append(eid[i])
        return edl, fl, itvl

            
    def panda(self):
        vall = self.show_value()
        edl,fl, itvl = self.find_edges_direction()
        df1 = pd.DataFrame(edl, columns = ["start","end"])
        df2 = pd.DataFrame(itvl, columns = ["starting coord","ending coord"])
        df = pd.concat([df1,df2],axis = 1)
        df["counting"] = [i+1 for i in range(df.shape[0])]
        start_list = []
        end_list = []
        for i in range(df.shape[0]):
            start_list.append(vall[df.iloc[i,0]])
            end_list.append(vall[df.iloc[i,1]])
        df["starting value"] = start_list
        df["ending value"] = end_list
        df["f"] = fl
        return df
    
    def solve(self,edge):
        dff = self.panda()
        df = dff[(dff['start'] == edge[0]) & (dff['end'] == edge[1])]
        TS, x, T = neumann_find_sol_bvp(df.iloc[0,7],df.iloc[0,2],df.iloc[0,3],df.iloc[0,5],df.iloc[0,6])
        return TS, x, T
    
    def plot(self,edge):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(self.solve(edge)[1], self.solve(edge)[2], '^g', label='Approximate solution')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$T$')
        ax.set_title('Solution for equation u\'\' = f')
        ax.legend();
        
    def plot_curve(self,edge):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(self.solve(edge)[1], self.solve(edge)[2], label='Approximate solution')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$T$')
        ax.set_title('Solution for equation u\'\' = f')
        ax.legend();
    
#    def solve_PDE_with_BV
# All above was great, so all we do here is to use the +u version of initial condition, so copy and change:
    def solve_lambda(self,edge):
        dff = self.panda()
        df = dff[(dff['start'] == edge[0]) & (dff['end'] == edge[1])]
        TS, x, T = neumann_find_sol_bvp_lambda(df.iloc[0,7],df.iloc[0,2],df.iloc[0,3],df.iloc[0,5],df.iloc[0,6])
        return TS, x, T
    
    def plot_lambda(self,edge):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(self.solve_lambda(edge)[1], self.solve_lambda(edge)[2], '^g', label='Approximate solution')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$T$')
        ax.set_title('Solution for equation u\'\' - u = f on the edge{}'.format(edge))
        ax.legend();
        
    def plot_curve_lambda(self,edge):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(self.solve_lambda(edge)[1], self.solve_lambda(edge)[2], label='Approximate solution')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$T$')
        ax.set_title('Solution for equation u\'\' - u = f on the edge{}'.format(edge))
        ax.legend();


# In[14]:


def d2_mat_dirichlet(nx, dx):
    """
    Constructs the centered second-order accurate second-order derivative for
    Dirichlet boundary conditions.

    Parameters
    ----------
    nx : integer
        number of grid points
    dx : float
        grid spacing

    Returns
    -------
    d2mat : numpy.ndarray
        matrix to compute the centered second-order accurate first-order deri-
        vative with Dirichlet boundary conditions on both side of the interval
    """
    # We construct a sequence of main diagonal elements,
    diagonals = [[1.], [-2.], [1.]]
    # and a sequence of positions of the diagonal entries relative to the main
    # diagonal.
    offsets = [-1, 0, 1]

    # Call to the diags routine; note that diags return a representation of the
    # array; to explicitly obtain its ndarray realisation, the call to .toarray()
    # is needed. Note how the matrix has dimensions (nx-2)*(nx-2).
    d2mat = diags(diagonals, offsets, shape=(nx-2,nx-2)).toarray()

    # Return the final array divided by the grid spacing **2.
    return d2mat / dx**2


# In[15]:


def neumann_find_sol_bvp(f,a,b,aval,bval):
    """
    Constructs the centered second-order accurate second-order derivative for
    Dirichlet boundary conditions. Equation is Laplacian(u) = f.
    We use 40 steps.

    Parameters
    ----------
    f: function on the edge
        type: function
    a : starting coordinate
        type: float
    b : ending coordinate
        type: float
    aval : starting value
        type: float
    bval : ending value
        type: float

    Returns
    -------
    TS: interpolated solution
        type: function
    x: grid points on the edge-interval
        type: list
    T : values of the solution on the edge-interval
        type: list
    """
    nx = 41                 # number of grid points
    lx = b-a                   # length of interval
    dx = lx / (nx-1)          # grid spacing
    x = np.linspace(a, b, nx) # coordinates of points on the grid
    b = f(x)      # right-hand side vector at the grid points
    T = np.empty(nx)          # array to store the solution vector
    # We use d2_mat_dirichlet() to create the skeleton of our matrix.
    
    A = d2_mat_dirichlet(nx, dx)

    # The first line and last line of A needs to be modified for the Neumann boundary condition.

    # Computation of the inverse matrix. The psudo-inverse is because if we use convergence degree 1, 
    #it is a singular matrix.
    A_inv = np.linalg.pinv(A)

    # Perform the matrix multiplication of the inverse with the right-hand side.
    # We only need the values of b at the interior nodes.
    b[1] = b[1] - aval/dx**2
    b[-2] = b[-2] - bval/dx**2
    T[1:-1] = np.dot(A_inv, b[1:-1])

    # Manually set the boundary values in the temperature array.
    T[0], T[-1] = [aval, bval]
    
    
    # Now we use interpolation to return the function of the computed T.
    TS = interp1d(x, T)

    return TS, x, T


# In[16]:


def neumann_find_sol_bvp_lambda(f,a,b,aval,bval):
    """
    Constructs the centered second-order accurate second-order derivative for
    Dirichlet boundary conditions. Equation is Laplacian(u) + u = f.
    We use 40 steps.

    Parameters
    ----------
    f: function on the edge
        type: function
    a : starting coordinate
        type: float
    b : ending coordinate
        type: float
    aval : starting value
        type: float
    bval : ending value
        type: float

    Returns
    -------
    TS: interpolated solution
        type: function
    x: grid points on the edge-interval
        type: list
    T : values of the solution on the edge-interval
        type: list
    """
    nx = 41                  # number of grid points
    lx = b-a                   # length of interval
    dx = lx / (nx-1)          # grid spacing
    x = np.linspace(a, b, nx) # coordinates of points on the grid
    b = f(x)      # right-hand side vector at the grid points
    T = np.empty(nx)          # array to store the solution vector
    # We use d2_mat_dirichlet() to create the skeleton of our matrix.
    
    A = d2_mat_dirichlet(nx, dx)

    # The first line and last line of A needs to be modified for the Neumann boundary condition.

    # Computation of the inverse matrix. The psudo-inverse is because if we use convergence degree 1, 
    #it is a singular matrix.
    identity = [[1.]]
    idt = diags(identity, [0], shape=(nx-2,nx-2)).toarray()
    A = A-idt
    A_inv = np.linalg.pinv(A)

    # Perform the matrix multiplication of the inverse with the right-hand side.
    # We only need the values of b at the interior nodes.
    b[1] = b[1] - aval/dx**2
    b[-2] = b[-2] - bval/dx**2
    T[1:-1] = np.linalg.solve(A,b[1:-1])

    # Manually set the boundary values in the temperature array.
    T[0], T[-1] = [aval, bval]
    
    
    # Now we use interpolation to return the function of the computed T.
    TS = interp1d(x, T)

    return TS, x, T

