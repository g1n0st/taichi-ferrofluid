 # taichi_ferrofluid
Implementation of *A Level-Set Method for Magnetic Substance Simulation* in the Taichi language.

**The fluid part is finished, magnetic force part is still an arduous work in progress.**

Running the Demo
----------------

This project is fully written in the Taichi Language, you can use the following commands to install the dependency:

```bash
python3 -m pip install taichi
python3 -m pip install taichi_glsl
```

The simulator kernels `{fluid_simulator.py|ferrofluid_simulator.py}` are designed dimensionality independent thanks to the features of Taichi. To initialize the scene, you should provide an initializer to the simulator, see `{initializer_2d.py|initializer_3d.py}` for details. Moreover, to visualize the result per frame, you should provide a visualizer, see `{visualizer_2d.py|visualizer_3d.py}` for details. You can see the examples demonstrated in `{main_2d.py|main_3d.py}`.

## Neat things about this implementation

Our reference paper is *A level-set method for magnetic substance simulation*, and most fluid simulation algorithms in Bridson's *Fluid Simulation for Computer Graphics*, 2nd ed. (2015).

#### Poisson Equation

One substep of the simulation needs to solve several Poisson equations: (1) solve potential function psi (2) solve pressure and apply magnetic force (3) solve surface tension (4) solve pressure again and apply volume control. To optimize the performance, we use the Multi-grid Preconditioned Conjugate Gradient scheme proposed in *A parallel multigrid Poisson solver for fluids simulation on large grids* and use a red-black Gauss-Seidel smooth operator for both time and space efficiency. It accepts a `Strategy` class to initialize the Ax=b coefficients and `cell_type` to indicate the fluid domain. Plus, we have tried the MIC(0) preconditioner and found MGPCG is way better in convergence.

See `{mgpcg.py}` for details.

#### Level-Set

The original paper uses the Semi-Lagrangian method to advect level-set and uses A Fast Marching Method to redistance it. However, its implementation needs a priority queue (see `{priority_queue.py}` for details) to get the nearest neighborhood, which it's hard to parallel in a regular grid. Instead, we used A Fast Sweeping Method to solve the Eikonal equation and verified its performance in the CUDA backend. Meanwhile, we implemented the algorithm that builds level-set from marker particles. 

See `{level_set.py}` for details.

#### Advection

The original paper uses the Semi-Lagrangian method to advect velocities and level-set. We implemented RK2 Semi-Lagrangian advection, markers advection, moreover, tried the APIC advection scheme (see `{apic_extension.py}` for details) to test the expansibility of this paper.

#### Volume Control

We found when applied surface forces (surface tension and magnetic force) or coupled with APIC, the volume of the fluid changed dramatically. The paper uses the Volume Control Method, which adds a divergence constant c as a volume correction term to compensate for the volume loss or gain. It works quite well to make the fluid incompressible.

See `{volume_control.py}` for details.

#### Ghost Fluid Method

The voxelized treatment of the free surface boundary condition is p = 0 in the empty cells. Even if we can track and render an accurate water surface, so far the core of the simulation - the pressure solve - only sees a block voxelized surface, which cannot avoid significant voxel artifacts. We followed the original paper  to use the ghost fluid method to make more accurate pressure solves.

See `{pressure_project.py}` for details.

#### Potential Function

It's doubtful that the right-hand side of the discretization formula (35) in the original paper might be wrong, that if the external H field is constant, the entire right hand-side will be zero. We discretize the Poisson Equation in another form, solve with a constant external magnetic field and a circular fluid domain, then visualize the potential function. It found that the result is the same as Fig. 7. in the Magnetic shielding boundary condition. It worth noting that choose a reference point r0 with psi(r0)=0 is essential, which makes the convergence way faster.

See `{potential_function.py}` for details.

References
----------------

[1] Xingyu Ni, Bo Zhu, Bin Wang, and Baoquan Chen. 2020. A level-set method for magnetic substance simulation. *ACM Trans. Graph.* 39, 4, Article 29 (July 2020), 15 pages. DOI:https://doi.org/10.1145/3386569.3392445

[2] James A. Sethian. 1996. A fast marching level set method for monotonically advancing fronts. *Proceedings of the National Academy of Sciences* 93, 4 (1996), 1591--1595.

[3] H. Zhao. A fast sweeping method for Eikonal equations. Math. Comp., 74:603–627, 2005.

[4] Wen Zheng, Jun-Hai Yong, and Jean-Claude Paul. 2006. Simulation of Bubbles. In *Proceedings of the 2006 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA '06).* Eurographics Association, Goslar, DEU, 325--333.

[5] *Byungmoon Kim, Yingjie Liu, Ignacio Llamas, Xiangmin Jiao, and Jarek Rossignac. 2007. Simulation of bubbles in foam with the volume control method. In* *ACM SIGGRAPH 2007 papers* *(**SIGGRAPH '07**). Association for Computing Machinery, New York, NY, USA, 98–es.*

[6] C. Jiang, C. Schroeder, A. Selle, J. Teran, and A. Stomakhin. 2015. The affine particle-in-cell method. ACM Trans Graph 34, 4 (2015), 51:1–51:10.

[7] F. Gibou, R. Fedkiw, L.-T. Cheng, and M. Kang. A second-order- accurate symmetric discretization of the Poisson equation on irreg- ular domains. J. Comp. Phys., 176:205–227, 2002.

[8] *A. McAdams, E. Sifakis, and J. Teran. 2010. A parallel multigrid Poisson solver for fluids simulation on large grids. In* *Proceedings of the 2010 ACM SIGGRAPH/Eurographics Symposium on Computer Animation* *(**SCA '10**). Eurographics Association, Goslar, DEU, 65–74.*

[9] *Robert Bridson. 2008.* Fluid Simulation. A. K. Peters, Ltd., USA.
