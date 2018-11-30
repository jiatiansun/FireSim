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

### What We Have Achieved:
As our fire simulation adapts general simulation of fluid motion via solving Navier-Stokes equations, we explore first simulation of more general fluid, specifically water(ocean). In  the exploration, we gained better understanding of computer graphics theory behind the particle simulation and have successfully created a short demo of simulation of ocean with processing:(Rough Ocean Simulation)[https://youtu.be/bxSOGtfZEEE]

### Expectation for Deliverables:
We still strongly believe that the fire simulation demo is deliverable by the poster session. As we dive deeper into the simulation algorithm, we gain more confidence in writing faster simulation code. As for the “nice to haves”, it is quite possible that we will generate plenty of simulation in the process of creating fire simulation demo. Specifically, so far, we have already created a small ocean simulation demo. These process demo videos can also be utilized in our final poster session to showcase the different functionality of the same simulation model. However, considering our current process, we are afraid that other “nice to haves” like beautiful user interface will be hard to be achieved before deadline. Instead, we will focus mainly on optimization of different fluid model in the following weeks to make sure that eventually a concrete fire simulation demo can be delivered.

### Plan for Poster Session
As what we planned before, we will showcase a demo of fire simulation in the poster session. We will also show a graph of the initial speedup of our program and its eventual speedup. Demo video of different kinds of fluid we would come through before the session might be displayed as well.

## PLATFORM CHOICE 
   Windows & CUDA. 
   Core computation will be done on GPU. Using GPU for this application makes sense since kernel functions for particle update are small, but there are many particles.
   Rendering will be done using an graphics API, potentially OpenGL.

## SCHEDULE Update	
   Nov 1st - Nov 19th
   Made different versions of toy demo with Processing. 
   Specifically, we accomplished ocean simulation by solving Navier Stroke , which can be later adapted to the simulation of fire.
   Nov 20th - Nov 23rd 
   Use Processing to experiment with simulation of other models like runnning river

   Nov 24th - Nov 27th  
   Use c++ to rewrite the CPU version of fire simulation. 

   Nov 18th - Nov 25th  
   Transfer the CPU version to GPU version using OpenGl - Caroline/Jackie
	
   Nov 25th - Nov 30th
   Experiment the GPU code with different hardwares, pick one hardware and start optimization based on the hardware. - Caroline/Jackie
   
   Nov 31st - Dec 14th  
   One person focuses on optimization of basic mechanism of particle rendering, while the other focuses on optimization of fire simulation specifically - Jackie/Caroline (The final week will reduce the time we have to work on the project)
   
   Dec 14th - Dec 16th 
   Creation of final Demo and materials for the poster session
