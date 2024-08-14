import gmsh
import os

def make_annulus(params):

    lc = params['lc']
    Ri = params['r0'] - params['delta']/2
    Re = params['r0'] + params['delta']/2
    
    gmsh.initialize()
    # Outer annulus arc points
    factory = gmsh.model.geo

    factory.addPoint(
        0, Re, 0, lc, 1 # (x, y, z, lc, tag)
    )

    factory.addPoint(Re, 0, 0, lc, 2)
    factory.addPoint(0, -Re, 0, lc, 3)

    # Inner annulus arc points
    factory.addPoint(0, -Ri, 0, lc, 4)
    factory.addPoint(Ri, 0, 0, lc, 5)
    factory.addPoint(0, Ri, 0, lc, 6)

    # Centre of the annulus
    factory.addPoint(0, 0, 0, lc, 7)

    # Implement the annulus arcs and line boundaries
    factory.addCircleArc(1, 7, 2, 1)
    factory.addCircleArc(2, 7, 3, 2)
    factory.addLine(3, 4, 3)
    factory.addCircleArc(4, 7, 5, 4)
    factory.addCircleArc(5, 7, 6, 5)
    factory.addLine(6, 1, 6)

    # Produce curve loop
    factory.addCurveLoop([1, 2, 3, 4, 5, 6], 7)
    factory.addPlaneSurface([7], 8)

    factory.synchronize()
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6], 9)
    gmsh.model.addPhysicalGroup(2, [8], 10)

    gmsh.model.mesh.generate(2)
    gmsh.write(os.path.join(params['output_dir'], 'mesh.msh'))



    gmsh.fltk.run()

    gmsh.finalize()
    
    return gmsh.model

def make_rectangle(params):

    lc = params['lc']
    Lx = params['Lx']
    Ly = params['Ly']
    
    gmsh.initialize()
    # Corners of the rectangle
    factory = gmsh.model.geo

    factory.addPoint(
        0, 0, 0, lc, 1 # (x, y, z, lc, tag)
    )
    factory.addPoint(Lx, 0, 0, lc, 2)
    factory.addPoint(Lx, Ly, 0, lc, 3)
    factory.addPoint(0, Ly, 0, lc, 4)

    factory.addLine(1, 2, 1)
    factory.addLine(2, 3, 2)
    factory.addLine(3, 4, 3)
    factory.addLine(4, 1, 4)

    # Produce curve loop
    factory.addCurveLoop([1, 2, 3, 4], 5)
    factory.addPlaneSurface([5], 6)

    factory.synchronize()
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 7)
    gmsh.model.addPhysicalGroup(2, [6], 8)

    gmsh.model.mesh.generate(2)
    gmsh.write(os.path.join(params['output_dir'], 'mesh.msh'))

    #gmsh.fltk.run()
    gmsh.finalize()

    return gmsh.model