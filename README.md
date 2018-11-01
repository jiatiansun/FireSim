## Summary
This project aims to render fire in real-time using physics-based methods. 
All particle computation will be done on GPU. Rendering will be done on GPU through the use of an graphics API (potentially OpenGL)

## BACKGROUND

The fire will be simulated through simulation of fluid motion via solving Navier-Stokes equations.
The space that’s being rendered will be discretized into cubical cells. Each grid cell is responsible for storing the physics state such as temperature, pressure, and velocity. The cells will be mapped onto individual SIMD cores.  In each time step, the states associated with each grid cell will be updated according to the Navier-Stokes equation.
This is a compute-intensive application, and will hugely benefit from parallel computation because the update scheme for individual cells is identical. This means that multiple cells can potentially be updated simultaneously with little divergence. 

## THE CHALLENGE
Dependencies between cells are the most challenging part. Since each cell needs to be updated according to the states of nearby cells, there is inherent sequential computation. 

## RESOURCES:
We want to implement Prof. Keenan Crane’s project on Real-Time Simulation and Rendering of 3D Fluids.  [GPUFluid](http://www.cs.cmu.edu/~kmcrane/Projects/GPUFluid/paper.pdf) There is detailed theory introduction as well as pseudo-code for simulation kernels.
The computer we have is an Alienware laptop with Intel i7-8750H (6core, 9MB 1st level cache) and NVIDIA GeForce 1070 GPU 
Jackie has written a toy engine that has basic camera operations implemented. We might start our rendering framework from there. 			
 
## GOALS AND DELIVERABLES
### PLAN TO ACHIEVE
Physically based simulation of fire. We expect the simulation to be done in real-time. Thus, rendering of one image needs to be rendered in milliseconds.
We expect to show an interactive demo of the fire simulation, in which a realistic fire burns for a duration of time inside the video. The user can freely rotate the camera to view the fire from different angles. If time allowed, the user can interactively change the position of the fire through mouse motion. Fire-resistant obstacles can additionally be inserted. 
We consider a physically-based realistic looking real-time generated demo will be considered as achieving the goal of the project from the performance perspective. We will demonstrate also a speedup graph to demonstrate that our project makes efficient use of our resources.
### HOPE TO ACHIEVE   
1. Make a more user-friendly graphical interface that can be used interactively in the poster session.
2. Use the solver to generate interactive smoke

## PLATFORM CHOICE 
   Windows & CUDA. 
   Core computation will be done on GPU. Using GPU for this application makes sense since kernel functions for particle update are small, but there are many particles.
   Rendering will be done using an graphics API, potentially OpenGL.

## SCHEDULE
1. Nov 1st - Nov 8th: Make toy demo with Processing
			(Midterm of our probability classes will slow us down a little bit)
2. Nov 9th - Nov 15rd: Make CPU version of the fire simulation with very few particles 
(Midterm of 418 will also speed us down significantly, so part of jobs will be pushed off to next week before the checkpoint deadline)
3. Nov 16th- Nov 23rd: Submit the project for the checkpoint and starts working on the GPU 
 version of the fire simulation
4. Nov 24th - Nov 30th: Accomplish GPU version of the fire simulation
5. Nov 31st - Dec 14th: Accomplish Performance Optimization of the fire simulation
			  (The final week will reduce the time we have to work on the project)
6. Dec 14th - Dec 16th: Creation of final Demo and materials for the poster session

