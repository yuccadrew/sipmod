# flake8: noqa
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 07:44:11 2022

@author: john775
"""

import numpy as np
import os
import glob


class element:
    def __init__(self):
        self.g_ind = -999  # global index for this element
        self.l_ind = -999  # local index for this elemnt
        self.g_nodes = np.zeros(3, dtype='int32')-1  # global node(vertex) numbers for this element
        self.l_nodes = np.zeros(3, dtype='int32')-1  # local node numbers for this element, probably won't be used
        self.l_neighs = np.zeros(3, dtype='int32')-1  # local neighbor element numbers for this element
        self.flags = np.zeros(3, dtype='int32')  # placeholder for element flags
        self.ghost = False  # True if this element is not owned by this process
        self.owner = -1  # process that owns this element
        self.props = np.zeros(4)  # properties of this element (i.e. conductivity etc.)

    def delete(self):
        del self


class node:
    def __init__(self, nd):
        self.x = -999.
        self.y = -999.
        self.z = -999.
        self.g_ind = -1  # row where this node is listed in the node mesh file
        self.petsc_inds = np.zeros(nd, dtype='int32')  # row positions for each dof
        self.ndof = nd  # number of degress of freedom for this node
        self.dof = np.zeros(nd, dtype=complex)  # value of each dof on this node
        self.flags = np.zeros(3, dtype='int32')  # node flags
        self.cons = np.empty(0, dtype='int32')  # holds indices of all non-zeros columns for this node ...
        # ... these indices correspond to the first dof for this node
        self.ghost = False

    def addcon(self, val):  # appends a connection to this node, including self
        self.cons = np.append(self.cons, val)

    def delete(self):
        del self


def allocate_AIJ_sparse(nnods, nstart, nend, dof, nods, ele, ghost_ele):
    # start by building the d_nnz and -_nnz vectors for the rows this process owns
    # see https://petsc.org/release/docs/manual/mat/
    nrows = nend-nstart+1
    d_nnz = np.zeros((dof*nrows), dtype='int32')
    o_nnz = np.zeros((dof*nrows), dtype='int32')

    # add the neighbor list for each node (including the node itself)
    for e in ele:
        for lnd in e.l_nodes:
            for gnd in e.g_nodes:
                if gnd not in nods[lnd].cons:
                    nods[lnd].addcon(gnd)

    for e in ghost_ele:
        for lnd in e.l_nodes:
            for gnd in e.g_nodes:
                if gnd not in nods[lnd].cons:
                    nods[lnd].addcon(gnd)

    # sort the connections
    for n in nods:
        n.cons.sort()

    cnt = 0
    for n in nods:
        if (not n.ghost):
            for c in n.cons:
                if ((c >= nstart) and (c <= nend)):
                    for i in range(dof*cnt, dof*(cnt+1)):
                        d_nnz[i] += 1
                else:
                    for i in range(dof*cnt, dof*(cnt+1)):
                        o_nnz[i] += 1
            cnt += 1

    # A = PETSc.Mat()
    # A.createAIJ(((nrows*dof, nnods*dof), (nnods*dof, nnods*dof)), comm=comm)
    # A.setISPreallocation(nnz=d_nnz, onnz=o_nnz)
    # return A
    return {'nnz': d_nnz, 'onnz': o_nnz}


def read_my_nodes(mesh_pre, my_rank, dof, ele, ghost_ele):
    # this subroutine reads the nodes (i.e. vertexes) belonging to this process
    # including the ghost nodes and other pertinant nodal information
    # It also build the global to local nodal mapping vector for this process
    import linecache
    linecache.clearcache()

    # read the node partitioning
    npart = np.genfromtxt(mesh_pre+'.node', skip_header=1, usecols=4, dtype='int')
    nnods = len(npart)

    # inds holds the global index of nodes owned by this process, and is a sequential
    # list of integers (because the mesh has been partitioned)
    inds = np.where(npart == my_rank)[0]

    # initialize a global to local node mapping vector
    nodemap = np.zeros(nnods, dtype='int32') - 1

    # read in the nodes and node properties
    mynods = []
    nmynods = len(inds)
    ni = 0
    for i in inds:
        mynods.append(node(dof))
        line = linecache.getline(mesh_pre+'.node', i+2).split()
        nodemap[i] = ni
        mynods[ni].x = float(line[1])
        mynods[ni].y = float(line[2])
        mynods[ni].g_ind = np.int32(i)
        for j in range(dof):
            mynods[ni].petsc_inds[j] = i*dof+j
        mynods[ni].ndof = dof
        mynods[ni].flags[0] = int(line[3])
        ni += 1

    # loop over the ghost elements to find an add ghost nodes
    nstart = inds[0]
    nend = inds[nmynods-1]

    for ge in ghost_ele:
        for nod in ge.g_nodes:
            if ((nod < nstart or nod > nend) & (nodemap[nod] == -1)):
                mynods.append(node(dof))
                line = linecache.getline(mesh_pre+'.node', nod+2).split()
                nodemap[nod] = ni
                mynods[ni].x = float(line[1])
                mynods[ni].y = float(line[2])
                mynods[ni].g_ind = nod
                for j in range(dof):
                    mynods[ni].petsc_inds[j] = nod*dof+j
                mynods[ni].ndof = dof
                mynods[ni].flags[0] = int(line[3])
                mynods[ni].ghost = True
                ni += 1

    # set the local nodes for each element and ghost element
    for e in ele:
        for j in range(3):
            e.l_nodes[j] = nodemap[e.g_nodes[j]]

    for e in ghost_ele:
        for j in range(3):
            e.l_nodes[j] = nodemap[e.g_nodes[j]]

    return nnods, nstart, nend, mynods


def read_my_elements(mesh_pre, my_rank):
    # subroutine reads the elements belonging to this processes
    # including the ghost elements and all pertinent element information

    import linecache
    linecache.clearcache()
    # read the element partioning
    epart = np.genfromtxt(mesh_pre+'.ele', usecols=(5), dtype='int', skip_header=1)

    # inds holds the global element indexes that belong to this process
    # note inds will be a sequential list of integers because the
    # mesh is partitioned
    inds = np.where(epart == my_rank)[0]

    # nmyele is the number of elements that belong to this process
    nmyele = len(inds)

    # read in the elments belonging to this process and the corresponding neighbors
    myele = np.genfromtxt(mesh_pre+'.ele', skip_header=inds[0]+1, max_rows=nmyele, usecols=(1, 2, 3, 4), dtype='int32')
    myele[:, 0:3] += -1
    myneighs = np.genfromtxt(mesh_pre+'.neigh', skip_header=inds[0]+1, max_rows=nmyele, usecols=(1, 2, 3), dtype='int32')
    myneighs += -1

    # initialize the elements that belong to this process
    # subtract 1 or zero base indexing
    ele = []
    for i in range(nmyele):
        ele.append(element())
        ele[i].g_ind = np.int32(inds[i])
        ele[i].l_ind = np.int32(i)
        ele[i].g_nodes = myele[i, 0:3]
        ele[i].l_neighs = myneighs[i, 0:3]
        ele[i].flags[0] = myele[i, 3]
        ele[i].owner = my_rank

    # make a list of ghost elements, which are elements whose global index's
    # are outside of the range of inds.
    ghost_list = []
    e_start = inds[0]
    e_end = inds[nmyele-1]
    for e in ele:
        for i in range(3):  # for each edge. this will be 4 faces for 3D (i.e. tetrahedrons)
            test_n = e.l_neighs[i]
            if (((test_n > -1) & (test_n < e_start)) | (test_n > e_end)):
                if test_n not in ghost_list:
                    ghost_list.append(test_n)

    ghost_list.sort()

    # now read in the ghost elements
    ghost_ele = []
    lind = 0
    for g in ghost_list:
        line = linecache.getline(mesh_pre+'.ele', g+2).split()
        if (int(line[0])-1 != g):
            print('!!!Error: Ghost elements are not lining up!!!')
        ghost_ele.append(element())
        ghost_ele[lind].g_ind = g
        ghost_ele[lind].l_ind = np.int32(lind)
        ghost_ele[lind].g_nodes = [int(line[1])-1, int(line[2])-1, int(line[3])-1]
        ghost_ele[lind].ghost = True
        ghost_ele[lind].flags[0] = int(line[4])
        ghost_ele[lind].owner = int(line[5])
        line = linecache.getline(mesh_pre+'.neigh', g+2).split()
        ghost_ele[lind].neighs = [int(line[1])-1, int(line[2])-1, int(line[3])-1]
        ghost_ele[lind].g_nodes = np.array(ghost_ele[lind].g_nodes, dtype='int32')
        ghost_ele[lind].neighs = np.array(ghost_ele[lind].neighs, dtype='int32')
        lind += 1

    # !!!NOTE: CODE TO SET THE PROPERTIES OF EACH ELEMENT AND
    # GHOST ELEMENT SHOULD GO HERE!!!

    # return the element and ghost element lists
    return ele, ghost_ele


def SetupMatrix(mesh_pre, ndof, my_rank):
    # read the partioning ... assumes mesh_pre is a partitioned mesh file
    npart = np.genfromtxt(mesh_pre+'.node', usecols=(4), dtype='int', skip_header=1)
    # nnods = len(npart)
    inds = np.where(npart == my_rank)[0]
    my_row_start = inds[0]
    my_row_end = max(inds)+1
    my_nnods = len(inds)

    # get the node positions and flags
    pos = np.genfromtxt(mesh_pre+'.node', usecols=(1, 2), skip_header=1)
    flags = np.genfromtxt(mesh_pre+'.node', usecols=(3), skip_header=1)
    # init and populate the nodes
    my_nodes = [node(ndof) for i in range(len(inds))]
    ia, ja = Build_Tri_Adj(mesh_pre, my_row_start, my_row_end)
    for i in range(my_nnods):
        my_nodes[i].nat_ind = inds[i]
        my_nodes[i].x = pos[inds[i], 0]
        my_nodes[i].y = pos[inds[i], 1]
        my_nodes[i].flags[0] = flags[inds[i]]
        for j in range(ndof):
            my_nodes[i].petsc_inds[j] = ndof*inds[i]+j

    ia, ja = Build_Tri_Adj(mesh_pre, my_row_start, my_row_end)
    for i in range(my_nnods):
        for j in range(ia[i], ia[i+1]):
            my_nodes[i].addcon(ndof * ja[j])

    return my_row_start, my_row_end, my_nodes
    # my_nnods, nnods, ia, ja = Build_Tri_Adj(mesh_pre, myrank):


def vtk_output(myrank, nodes, elements, ghost_elements):
    import evtk

    # build the vert positions
    x = np.zeros(len(nodes))
    y = np.zeros(len(nodes))
    z = np.zeros(len(nodes))

    for i in range(len(nodes)):
        x[i] = nodes[i].x
        y[i] = nodes[i].y

    # build the connections, element offsets, and element type vectors
    # ele = np.genfromtxt(meshpre+'.ele', usecols=(1, 2, 3), skip_header=1)-1
    cons = np.zeros(3*(len(elements)+len(ghost_elements)))
    ofset = np.zeros(len(elements)+len(ghost_elements))
    ccnt = 0
    ecnt = 0
    for e in elements:
        ofset[ecnt] = 3*(ecnt+1)
        ecnt = ecnt+1
        for i in range(3):
            cons[ccnt] = e.l_nodes[i]
            ccnt = ccnt+1

    for e in ghost_elements:
        ofset[ecnt] = 3*(ecnt+1)
        ecnt = ecnt+1
        for i in range(3):
            cons[ccnt] = e.l_nodes[i]
            ccnt = ccnt+1

    ctype = np.zeros(len(elements)+len(ghost_elements))
    ctype[:] = evtk.vtk.VtkTriangle.tid

    # build the element and node data vectors
    cd = np.zeros(len(elements)+len(ghost_elements))-999
    cnt = 0
    for e in elements:
        cd[cnt] = float(e.ghost)
        cnt = cnt+1
    for e in ghost_elements:
        cd[cnt] = float(e.ghost)
        cnt = cnt+1

    cdata = {"Element Ghosts": cd}

    nd = np.zeros(len(nodes))-999
    cnt = 0
    for n in nodes:
        nd[cnt] = float(n.ghost)
        cnt = cnt+1

    pdata = {"Node Ghosts": nd}

    oname = 'rank'+str(myrank)
    evtk.hl.unstructuredGridToVTK(oname, x, y, z, connectivity=cons, offsets=ofset, cell_types=ctype, cellData=cdata, pointData=pdata)


def Create_Node_IS(mesh_pre, nrank, my_rank):
    try:
        from petsc4py import PETSc
        print('PETSc was imported')
    except ImportError:
        print('Failed to import PETSc')
        return

    # isg contains for each local node the global number of that node
    # isp contains the process number (rank) that each local node has been assigned to
    isg = PETSc.IS().create()
    isp = PETSc.IS().create()

    vals = np.genfromtxt(mesh_pre+'.node', skip_header=1, usecols=(4), dtype='int32')
    inds = np.where(vals == my_rank)
    isg.setIndices(inds[0].astype(np.int32))
    inds2 = np.copy(inds[0])
    inds2[:] = 0
    isp.setIndices(inds2.astype(np.int32))
    return isg, isp


def Build_Tri_Adj(mesh_pre, my_start, my_end):
    # my_start is the first row (or node) I'm responsible for
    # my_end is the last row I'm responsible fore

    class AdjRow:
        def __init__(self, n):
            self.n = n
            self.rstart = np.zeros([n+1, 1], dtype='int32')
            self.cinds = np.zeros([n, 100], dtype='int32')-999

    # load the triangle elements as node indexes
    ele = np.genfromtxt(mesh_pre+'.ele', skip_header=1, usecols=(1, 2, 3, 5), dtype='int32')
    ele = ele-1
    nele = len(ele)
    # nnods = np.amax(ele)+1

    # divide the nodes among ranks
    # div = int(np.floor(nnods/nrank))
    # my_start=myrank*div
    # my_end = (myrank+1)*div
    # if(myrank==nrank-1):
    #     my_end=nnods
    my_nnods = my_end-my_start

    ADJ = AdjRow(my_nnods)
    for i in range(nele):
        # search over each of the 3 nodes to see if they're already
        # included in each row
        for row in ele[i, 0:3]:
            if ((row >= my_start) & (row < my_end)):
                for col in ele[i, 0:3]:
                    inc = True
                    # check to see if this column has already been added
                    for cval in ADJ.cinds[row-my_start]:
                        if (cval == col):
                            inc = False
                            break
                    if (inc):
                        ADJ.cinds[row-my_start, ADJ.rstart[row-my_start]] = col
                        ADJ.rstart[row-my_start] += 1

    # sort the indexes
    for row in range(my_nnods):
        ADJ.cinds[row, 0:ADJ.rstart[row][0]] = np.sort(ADJ.cinds[row, 0:ADJ.rstart[row][0]])

    for i in range(my_nnods, 0, -1):
        ADJ.rstart[i] = np.sum(ADJ.rstart[0:i])
    ADJ.rstart[0] = 0

    ja = np.zeros((ADJ.rstart[my_nnods]), dtype='int32')
    ia = np.zeros(my_nnods+1, dtype='int32')
    cnt = 0
    for r in range(my_nnods):
        ia[r] = ADJ.rstart[r][0]
        for j in range(0, ADJ.rstart[r+1][0]-ADJ.rstart[r][0]):
            ja[cnt] = ADJ.cinds[r, j]
            cnt = cnt+1
    ia[my_nnods] = ADJ.rstart[my_nnods][0]
    return ia, ja


def LoadTriPlex(mesh_pre):
    try:
        from petsc4py import PETSc
        print('PETSc was imported')
    except ImportError:
        print('Failed to import PETSc')
        return

    # load the triangle elements
    ele = np.genfromtxt(mesh_pre+'.elee', skip_header=1, usecols=(1, 2, 3, 4), dtype='int32')
    ele = ele-1         # change to zero-based indexing
    nele = len(ele)

    # load the nodes (i.e. mesh vertices)
    nods = np.genfromtxt(mesh_pre+'.node', skip_header=1, usecols=(1, 2, 3, 4), dtype='float32')
    nnods = len(nods)

    # load the edges
    edges = np.genfromtxt(mesh_pre+'.edge', skip_header=1, usecols=(1, 2, 3), dtype='int32')
    edges = edges-1     # change to zero based indexing
    nedges = len(edges)

    nchart = nele+nnods+nedges

    # initialize the dmplex
    dm = PETSc.DMPlex().create()
    dm.setChart(0, nchart)

    # set the cone sizes for the elements
    for i in range(0, nele):
        dm.setConeSize(i, 3)
        # print(i, 3)
    # set the cone sizes for the edges
    for i in range(nele+nnods, nchart):
        dm.setConeSize(i, 2)
        # print(i, 2)
    # set up  the dmplex
    dm.setUp()

    # set the connections (i.e. the cone) for each element
    ele = ele+nele+nnods
    for i in range(0, nele):
        dm.setCone(i, ele[i, 0:3])
        # print(i, ele[i, 0:3])
    # set the connections (i.e. the cone) for the edges
    edges = edges+nele
    oset = nele+nnods
    for i in range(0, nedges):
        dm.setCone(i+oset, edges[i, 0:2])
        # print(i+oset, edges[i, 0:2])
    dm.symmetrize()
    dm.stratify()

    return dm


def metis_partition_tri(meshpre, npart,
                        ptype: str = 'none',
                        clean_files: bool = True):
    # Read the elements file and build the metis mesh input file
    f1 = open(meshpre+'.1.ele', 'r')
    line = f1.readline().split()
    f1.close()
    verts_per_ele = int(line[1])
    if verts_per_ele == 3:
        ele = np.genfromtxt(meshpre+'.1.ele', skip_header=1, usecols=(1, 2, 3, 4), dtype='int')
        nodes = np.genfromtxt(meshpre+'.1.node', skip_header=1, usecols=(1, 2, 3))
    else:
        print('Wrong number of vertices in the element file for a 2D mesh.')
        return

    np.savetxt(meshpre+'.msh', ele[:, 0:3], header=str(len(ele)), fmt='%-10.0f', comments='')

    if npart > 1:  # metis won't run for npart=1
        # call metis ... metis must be callable from the command line
        if ptype in ['kway', 'rb']:
            os.system('mpmetis '+meshpre+'.msh '+str(npart)+' -ptype='+ptype)
        else:
            os.system('mpmetis '+meshpre+'.msh '+str(npart))
    else:
        np.savetxt(
            meshpre+'.msh.npart.'+str(npart),
            np.zeros(len(nodes)),
            fmt='%.0f'
        )
        np.savetxt(
            meshpre+'.msh.epart.'+str(npart),
            np.zeros(len(ele)),
            fmt='%.0f'
        )

    # build the mapping vectors to re-order the mesh
    nodpart = np.genfromtxt(meshpre+'.msh.npart.'+str(npart), dtype='int')
    elepart = np.genfromtxt(meshpre+'.msh.epart.'+str(npart), dtype='int')

    nodmap = np.zeros((len(nodpart), 2), dtype='int')-1
    nodmap[:, 0] = nodpart

    # build the node mapping vector
    cnt = 0
    for part in range(npart):
        for i in range(len(nodpart)):
            if ((nodmap[i, 0] == part) & (nodmap[i, 1] == -1)):
                nodmap[i, 1] = cnt
                cnt = cnt+1
    nodorder = np.argsort(nodmap[:, 1])

    # reorder and write a new node file
    # nodes = np.genfromtxt(meshpre+'.1.node', skip_header=1, usecols=(1, 2, 3))
    f1 = open(meshpre+'.2.node', 'w')
    f2 = open(meshpre+'.2.nodpart', 'w')
    f1.write('{0:10.0f} 2 0 1 1\n'.format(len(nodorder)))
    f2.write(str(len(nodorder))+' 1\n')
    cnt = 1
    for row in nodorder:
        x = nodes[row, 0]
        y = nodes[row, 1]
        c1 = nodes[row, 2]
        c2 = nodmap[row, 0]
        f1.write('{0:10.0f} {1:20.12f} {2:20.12f} {3:5.0f} {4:5.0f}\n'.format(cnt, x, y, c1, c2))
        f2.write('{0:8.0f}\n'.format(c2))
        cnt = cnt+1
    f1.write('#Reordered node file constructed from mpart.py')
    f1.close()
    f2.close()

    # Correct the element indices for the new node ordering
    # ele = np.genfromtxt(meshpre+'.1.ele', skip_header=1, usecols=(1, 2, 3, 4), dtype='int')
    for e in ele:
        for i in range(3):
            e[i] = nodmap[e[i]-1, 1]+1

    # reorder and rewrite the element file
    elemap = np.zeros((len(elepart), 2), dtype='int')-1
    elemap[:, 0] = elepart
    cnt = 0
    for part in range(npart):
        for i in range(len(elepart)):
            if ((elemap[i, 0] == part) & (elemap[i, 1] == -1)):
                elemap[i, 1] = cnt
                cnt = cnt+1

    eleorder = np.argsort(elemap[:, 1])
    f1 = open(meshpre+'.2.ele', 'w')
    f2 = open(meshpre+'.2.elepart', 'w')
    f1.write('{0:10.0f} 3 1 1 \n'.format(len(eleorder)))
    f2.write(str(len(eleorder))+' 1\n')
    cnt = 1
    for row in eleorder:
        a = ele[row, 0]
        b = ele[row, 1]
        c = ele[row, 2]
        d = ele[row, 3]
        e = elemap[row, 0]
        f1.write('{0:10.0f} {1:10.0f} {2:10.0f} {3:10.0f} {4:10.0f} {5:8.0f}\n'.format(cnt, a, b, c, d, e))
        f2.write('{0:8.0f}\n'.format(e))
        cnt = cnt+1
    f1.write('#Reordered element file constructed from mpart.py')
    f1.close()
    f2.close()

    # Correct the edge indices for the new node ordering
    tmp = np.genfromtxt(meshpre+'.1.edge', skip_header=1, usecols=(0, 1, 2, 3), dtype='int')
    edge = np.zeros((len(tmp), 5), dtype='int')
    edge[:, 0:4] = tmp
    for e in edge:
        e[4] = nodmap[e[1]-1, 0]  # set the partition for this edge to partition of its first node
        for i in range(1, 3):
            e[i] = nodmap[e[i]-1, 1]+1  # set the new node numbers for this edge
    # edge = edge[edge[:, 3] > 0, :]  # only keep edges with flags greater than 0
    np.savetxt(meshpre+'.2.edge', edge, header=str(len(edge))+' 2', comments='', fmt='%-10.0f')

    # Correct the edge indices for the new face ordering
    # face = np.genfromtxt(meshpre+'.1.face',skip_header=1,dtype='int')
    # for e in face:
    #     for i in range(1,4):
    #         e[i] = nodmap[e[i]-1,1]+1
    # np.savetxt(meshpre+'.2.face',face,header=str(len(face))+' 1',comments='',fmt='%-10.0f')

    # reorder and rewrite the neighbor
    neigh = np.genfromtxt(meshpre+'.1.neigh', dtype='int', skip_header=1)
    for e in neigh:
        for i in range(1, 4):
            if e[i] > 0:
                e[i] = elemap[e[i]-1, 1]+1
    f1 = open(meshpre+'.2.neigh', 'w')
    f1.write(str(len(neigh))+' 3\n')
    cnt = 0
    for i in eleorder:
        cnt = cnt+1
        f1.write('{0:10.0f} {1:10.0f} {2:10.0f} {3:10.0f} \n'.format(cnt, neigh[i, 1], neigh[i, 2], neigh[i, 3]))
    f1.close()

    if clean_files:
        file_list = [meshpre+'.2.elepart', meshpre+'.2.nodpart']
        file_list = file_list + glob.glob(meshpre+'.msh*')
        for file_path in file_list:
            try:
                os.remove(file_path)
            except OSError:
                pass


def metis_partition_tet(meshpre, npart):
    # Read the elements file and build the metis mesh input file
    f1 = open(meshpre+'.1.ele', 'r')
    line = f1.readline().split()
    f1.close()
    verts_per_ele = int(line[1])
    if verts_per_ele == 3:
        ele = np.genfromtxt(meshpre+'.1.ele', skip_header=1, usecols=(1, 2, 3), dtype='int')
    if verts_per_ele == 4:
        ele = np.genfromtxt(meshpre+'.1.ele', skip_header=1, usecols=(1, 2, 3, 4), dtype='int')

    np.savetxt(meshpre+'.msh', ele, header=str(len(ele)), fmt='%-10.0f', comments='')

    # call metis ... metis must be callable from the command line
    os.system('mpmetis '+meshpre+'.msh '+str(npart))

    # build the mapping vectors to re-order the mesh
    nodpart = np.genfromtxt(meshpre+'.msh.npart.'+str(npart), dtype='int')
    elepart = np.genfromtxt(meshpre+'.msh.epart.'+str(npart), dtype='int')

    nodmap = np.zeros((len(nodpart), 2), dtype='int')-1
    nodmap[:, 0] = nodpart

    # build the node mapping vector
    cnt = 0
    for part in range(npart):
        for i in range(len(nodpart)):
            if ((nodmap[i, 0] == part) & (nodmap[i, 1] == -1)):
                nodmap[i, 1] = cnt
                cnt = cnt+1
    nodorder = np.argsort(nodmap[:, 1])

    # reorder and write a new node file
    nodes = np.genfromtxt(meshpre+'.1.node', skip_header=1, usecols=(1, 2, 3, 4, 5))

    f1 = open(meshpre+'.2.node', 'w')
    f2 = open(meshpre+'.2.nodpart', 'w')
    f1.write('{0:10.0f} 3 1 1 1\n'.format(len(nodorder)))
    f2.write(str(len(nodorder))+' 1\n')
    cnt = 1
    for row in nodorder:
        x = nodes[row, 0]
        y = nodes[row, 1]
        z = nodes[row, 2]
        c1 = nodes[row, 3]
        c2 = nodes[row, 4]
        c3 = nodmap[row, 0]
        f1.write('{0:10.0f} {1:20.12f} {2:20.12f} {3:20.12f} {4:5.0f} {5:5.0f} {6:8.0f}\n'.format(cnt, x, y, z, c1, c2, c3))
        f2.write('{0:8.0f}\n'.format(c3))
        cnt = cnt+1
    f1.write('#Reordered node file constructed from mpart.py')
    f1.close()
    f2.close()

    # Correct the element indices for the new node ordering
    ele = np.genfromtxt(meshpre+'.1.ele', skip_header=1, usecols=(1, 2, 3, 4, 5), dtype='int')
    for e in ele:
        for i in range(4):
            e[i] = nodmap[e[i]-1, 1]+1

    # reorder and rewrite the element file
    elemap = np.zeros((len(elepart), 2), dtype='int')-1
    elemap[:, 0] = elepart
    cnt = 0
    for part in range(npart):
        for i in range(len(elepart)):
            if ((elemap[i, 0] == part) & (elemap[i, 1] == -1)):
                elemap[i, 1] = cnt
                cnt = cnt+1

    eleorder = np.argsort(elemap[:, 1])
    f1 = open(meshpre+'.2.ele', 'w')
    f2 = open(meshpre+'.2.elepart', 'w')
    f1.write('{0:10.0f} 4 1 1 \n'.format(len(eleorder)))
    f2.write(str(len(eleorder))+' 1\n')
    cnt = 1
    for row in eleorder:
        a = ele[row, 0]
        b = ele[row, 1]
        c = ele[row, 2]
        d = ele[row, 3]
        e = ele[row, 4]
        f = elemap[row, 0]
        f1.write('{0:10.0f} {1:10.0f} {2:10.0f} {3:10.0f} {4:10.0f} {5:8.0f} {6:8.0f}\n'.format(cnt, a, b, c, d, e, f))
        f2.write('{0:8.0f}\n'.format(f))
        cnt = cnt+1
    f1.write('#Reordered element file constructed from mpart.py')
    f1.close()
    f2.close()

    # Correct the edge indices for the new node ordering
    edge = np.genfromtxt(meshpre+'.1.edge', skip_header=1, usecols=(0, 1, 2, 3), dtype='int')
    for e in edge:
        for i in range(1, 3):
            e[i] = nodmap[e[i]-1, 1]+1
    np.savetxt(meshpre+'.2.edge', edge, header=str(len(edge))+' 1', comments='', fmt='%-10.0f')

    # Correct the edge indices for the new face ordering
    face = np.genfromtxt(meshpre+'.1.face', skip_header=1, dtype='int')
    for e in face:
        for i in range(1, 4):
            e[i] = nodmap[e[i]-1, 1]+1
    np.savetxt(meshpre+'.2.face', face, header=str(len(face))+' 1', comments='', fmt='%-10.0f')

    # reorder and rewrite the neigh file
    neigh = np.genfromtxt(meshpre+'.1.neigh', dtype='int', skip_header=1)
    for e in neigh:
        for i in range(1, 5):
            if e[i] > 0:
                e[i] = elemap[e[i]-1, 1]+1
    np.savetxt(meshpre+'.2.neigh', neigh, header=str(len(neigh))+' 4', comments='', fmt='%-10.0f')
