#!/usr/bin/env python

import numpy as np
from numpy import sqrt

def placementKey( geo):
    """
    Given the positions of a list of the indices, create a unique key
    to register the position.
    """
    def diagcmp( xyA, xyB):
        """
        Compare two positions based on x + y. If x + y is the same for the
        two, compare based on x.
        """
        return cmp(xyA[0] + xyA[1], xyB[0] + xyB[1]) or cmp(xyA[0], xyB[0])

    sorted = [ tuple(geo[i]) for i in xrange(geo.shape[0]) ]
    sorted.sort( diagcmp)
    return hash(tuple(sorted))

class LonposPiece:
    """
    The geometric representation of a Lonpos piece, which is described
    by the 2-dimensional coordinates of each of the vertices. 
    """

    WHITE3  = [ [1,0], [0,0], [0,1] ]
    GREEN4  = [ [0,0], [1,0], [1,1], [0,1] ]
    ORANGE4 = [ [1,0], [0,0], [0,1], [0,2] ]
    PURPLE4 = [ [0,0], [0,1], [0,2], [0,3] ]
    BLUE5   = [ [1,0], [0,0], [0,1], [0,2], [0,3] ]
    CYAN5   = [ [2,0], [1,0], [0,0], [0,1], [0,2] ]
    GREEN5  = [ [0,0], [0,1], [0,2], [1,2], [1,3] ]
    RED5    = [ [0,0], [0,1], [0,2], [1,1], [1,2] ]
    PURPLE5 = [ [0,0], [0,1], [1,1], [1,2], [2,2] ]
    YELLOW5 = [ [0,0], [0,1], [1,1], [2,1], [2,0] ]
    GRAY5   = [ [0,1], [1,1], [1,0], [1,2], [2,1] ]
    PINK5   = [ [0,0], [0,1], [1,1], [0,2], [0,3] ]
    
    def __init__( self, geo, index=None):
        """
        For the given geometry, construct the symmetry and non-degenerate
        operations associated with the piece.
        """
        # the column vector self.geo[:,i] gives the i'th vertex's positions
        self.geo = np.array( geo, dtype=int)
        self.findNondegeneratePlacements()
        self.id = index
        
    @property
    def nVertices( self):
        return len(self.geo)

    @property
    def symbol( self):
        return chr( ord('a') + self.id) if not self.id is None else '*'
    
    def place( self, position, v, T):
        """
        Place the vertex v at position, and apply transformation T. Return
        the grid points that are occupied by the piece.
        """
        geo = (self.geo - self.geo[v]).dot( T)
        return position + geo
        
    def findNondegeneratePlacements( self):
        """
        Generate all non-degenerate placements, with one of the vertices placed
        at (0,0). Return the placements as [ (v, T) ], where v is the vertex
        to be placed at (0,0), and T the 2x2 transformation matrix that place
        the piece according to

        self.geo[v] + T.dot(self.geo - self.geo[v])
        """
        # Rotate counterclockwise by 90 degrees around the v'th vertex.
        r90 = np.array( [ [0,1], [-1,0] ], dtype=int)
        # Flip the piece along the vertical axis through the v'th vertex.
        fv = np.array( [ [1,0], [0,-1] ], dtype=int)

        self.placements = []
        uniques = set() # Unique placements generated so far
        identity = np.array( [ [1,0], [0,1] ], dtype=int)
        T = identity[:,:]
        for i in xrange(self.nVertices):
            geo = self.geo[:,:]
            geo -= geo[i] # Place i'th vertex at (0,0)
            for r in xrange(4):
                T = T.dot( r90)
                for f in xrange(2):
                    T = T.dot( fv)
                    pk = placementKey( geo.dot(T))
                    if (not pk in uniques):
                        uniques.add( pk)
                        self.placements.append( (i, T))
            # After four rotations and two flips, we should be back to
            # the original position.
            assert( np.array_equal( T, identity))

        return self.placements
                
class LonposBoard:
    """
    Representation of the Lonpos board.
    """

    AREA = 55
    
    def __init__( self, geo='triangle'):
        if (geo.upper()=='triangle'.upper()):
            L = int( (sqrt( 1 + 8*LonposBoard.AREA ) - 1)/2)
            self.grids = dict()
            self.positions = []
            for i in xrange(L):
                for j in xrange(L-i):
                    self.grids[(i,j)] = len(self.positions)
                    self.positions.append( (i,j))
        elif (geo.upper()=='rectangle'.upper()):
            self.grids = dict()
            self.positions = []
            X = 5
            Y = 11
            for i in xrange(X):
                for j in xrange(Y):
                    self.grids[(i,j)] = len(self.positions)
                    self.positions.append( (i,j))
                               
                    
class LonposState:
    """
    Representation of the state of the game, i.e., what pieces are placed on
    what positions of the board.
    """
    def __init__( self, board, occupation=[]):
        """
        Construct a Lonpos state, with the given board and pieces. The
        occupation array indicates which points of the board are occupied,
        and by what pieces. It is specified by [ (p, i) ], which indicate
        point p of the board is occupied by i'th piece. p can be either the
        index or the 2-d coordinates of the point.
        """
        self.board = board

        if (occupation):
            if (isinstance(occupation[0][0], int)):
                self.occupation = dict(occupation)
            elif (isinstance(occupation[0][0], tuple) and len(occupation[0][0])==2):
                try:
                    self.occupation = dict(
                        [ (board.grids[o[0]], o[1]) for o in occupation ])
                except KeyError, e:
                    raise e, "Occupied point not on board."
        else:
            self.occupation = dict(occupation)

    def show( self):
        """
        Use ASCII to illustrate the state of the Lonpos.
        """
        def symbol( i):
            return i<0 and (i==-2 and ' ' or '0') or chr(ord('a') + i)
        
        X, Y = np.max( self.board.positions, 0)
        # -2 to indicate outside board.
        display = np.zeros( (X+1,Y+1), dtype=int) - 2 
        for x, y in self.board.positions:
            display[x, y] = -1 # -1 to indicate unoccupied
        for p, i in self.occupation.items():
            x, y = self.board.positions[p]
            display[x, y] = i
        for x in xrange(X+1):
            s = ''.join( [ symbol( display[x, y]) for y in xrange(Y+1) ])
            print s
        
def placePiece( board, occupation, position, piece):
    '''
    Iterate through all permissible, non-degenerate, placements of the
    given piece such that the given position will be occupied, and fit
    within the current state of occupation of the board. Yield
    an iterator of the new state corresponding to each different
    placement.
    '''
    for p in piece.placements:
        vrts = piece.place( position, p[0], p[1])
        occupy = dict() # Occupation by the piece
        for v in vrts:
            ptid = board.grids.get( (v[0],v[1]), None)
            if (ptid is None): break # Outside the board
            if (occupation.has_key( ptid)): break # point already occupied
            occupy[ptid] = piece.id
        else:
            occupy.update( occupation)
            yield occupy

def countFreeNeighbors( p, board, occupation):
    """
    Count unoccupied neighbors of a point.
    """
    n = 0
    for m in [0, 1]:
        for d in [-1, 1]:
            pn = [p[0], p[1]]
            pn[m] += d
            j = board.grids.get( tuple(pn), None)
            if (j is None): continue # Not a board point
            if (occupation.has_key( j)): continue # Occupied
            n += 1
    return n
    
def findUnoccupied( board, occupation):
    """
    Find unoccupied positions on the board.
    """
    return [ j for j in xrange(len(board.positions))
             if not occupation.has_key(j) ]

def solve( board, pieces, occupation):
    """
    Use a depth-first-search to solve the Lonpos puzzle. 
    """

    from heapq import heappush, heappop

    unoccupied = findUnoccupied( board, occupation)
    remainingpieces = range(len(pieces))

    searchq = []
    nbacktrack = 0

    while (unoccupied):
        nnheap = []
        # As a heuristic, we choose to first place pieces on points
        # with the least number of unoccupied neighbors.
        for i in unoccupied:
            p = board.positions[i]
            nn = countFreeNeighbors( p, board, occupation)
            heappush( nnheap, (nn, i))
        nn, pt = heappop( nnheap)
        if (nn==0): # No solution, back-track
            if (searchq):
                occupation, remainingpieces = searchq.pop()
                nbacktrack += 1
                print "Backtracking for the %d'th time" % nbacktrack
                unoccupied = findUnoccupied( board, occupation)
                continue
            else:
                break
        for ipc in remainingpieces:
            pc = pieces[ipc]
            for o in placePiece( board, occupation, board.positions[pt], pc):
                # A search node is defined by the occupation state and
                # the remaining pieces.
                searchq.append( (o, [i for i in remainingpieces if i != ipc]))
        if (searchq):
            occupation, remainingpieces = searchq.pop()
            unoccupied = findUnoccupied( board, occupation)
        else:
            break
    else:
        state = LonposState( board, occupation.items())
        state.show()
        return occupation

    # No solution for the state.
    print "No solution!"
    return None  

import optparse
import sys

def unitTest():
    pieces = [
        LonposPiece( LonposPiece.WHITE3),   # 0
        LonposPiece( LonposPiece.GREEN4),   # 1
        LonposPiece( LonposPiece.ORANGE4),  # 2
        LonposPiece( LonposPiece.PURPLE4),  # 3
        LonposPiece( LonposPiece.BLUE5),    # 4
        LonposPiece( LonposPiece.CYAN5),    # 5
        LonposPiece( LonposPiece.GREEN5),   # 6
        LonposPiece( LonposPiece.RED5),     # 7
        LonposPiece( LonposPiece.PURPLE5),  # 8
        LonposPiece( LonposPiece.YELLOW5),  # 9
        LonposPiece( LonposPiece.GRAY5),    # 10
        LonposPiece( LonposPiece.PINK5)     # 11
        ]

    for i, p in enumerate(pieces): p.id = i

    def testTriangularBoard():
        board = LonposBoard()

        occupy = [
            ((0,0),19), ((0,1),19), ((1,0),19), ((2,0),19), ((2,1),19),
            ((3,0),13), ((3,1),13), ((3,2),13), ((3,3),13),
            ((4,0),15), ((4,1),15), ((4,2),15), ((5,0),15), ((6,0),15) ]

        state = LonposState( board, occupy)
        state.show()

        def testPlacements( board, state, pc, name):
            print "# %s: %d non-degenerate placements" % \
                  (name, len(pc.placements))
            for p in pc.placements:
                print p[0]
                print p[1]

            for o in placePiece( board, state.occupation, (5,2), pc):
                state = LonposState( board, o.items())
                state.show()

        for i, p in enumerate(pieces):
            testPlacements( board, state, p, "PIECE %d" % i)

        occupation = solve( board, [ pieces[i] for i in xrange(len(pieces))
                                     if i not in set([3,5,9]) ],
                            state.occupation)

    def testRectangleBoard():
        board = LonposBoard("rectangle")
        occupy = [
            ((0,0),0), ((0,1),0), ((1,1),0),
            ((1,5),1), ((1,6),1), ((2,5),1), ((2,6),1),
            ((2,7),2), ((3,7),2), ((3,8),2), ((3,9),2),
            ((4,0),3), ((4,1),3), ((4,2),3), ((4,3),3),
            ((0,2),4), ((1,2),4), ((0,3),4), ((0,4),4), ((0,5),4),
            ((1,0),5), ((2,0),5), ((3,0),5), ((3,1),5), ((3,2),5),
            ((2,1),6), ((2,2),6), ((2,3),6), ((1,3),6), ((1,4),6),
            ((0,8),7), ((0,9),7), ((0,10),7), ((1,9),7), ((1,10),7),
            ((0,6),8), ((0,7),8), ((1,7),8), ((1,8),8), ((2,8),8),
            ((2,9),9), ((2,10),9), ((3,10),9), ((4,10),9), ((4,9),9),
            ((3,3),10), ((3,4),10), ((2,4),10), ((4,4),10), ((3,5),10),
            ((4,5),11), ((4,6),11), ((3,6),11), ((4,7),11), ((4,8),11)
            ]

        leaveout = [2, 4, 7, 8]
        occupy = [ o for o in occupy if not o[1] in set(leaveout) ]
        
        state = LonposState( board, occupy)
        state.show()

        occupation = solve( board, [ pieces[i] for i in leaveout ],
                            state.occupation)
        
    testTriangularBoard()
    # testRectangleBoard()
    
if __name__ == '__main__':
    usage = '%prog state.in'
    opt = optparse.OptionParser( usage)
    opt.add_option( '--unit-test', action='store_true', default=False)
    opts, args = opt.parse_args()

    if (opts.unit_test):
        unitTest()
        sys.exit()

    pass 
