# SelfdrivingCars
 Implementation of a genetic algorithm in order to produce smart cars that can complete a simple track.

# How it works
A population of cars tries to complete a user specified track and earn a fitness score according to how far along the track they travelled. After a set amount of time, a new generation of cars is generated. Parents for the new generation are selected based on their fitness. Random mutations are introduced into the new generation in order to adapt to the environment.

## Set Up
Before starting, you can specify some parameters in `constants.py`.
* N_Sensors: number of sensors that the cars use to navigate.
* TRACKWIDTH: width of the track you are drawing.
* T_MAX: how many timesteps each generation will have to complete the track.
* PopulationSize: number of cars per generation (increasing the number too much will slow down the program).
* mutationRate, changeRate: parameters that control amount and severity of mutational changes in subsequent generations.
* maxFOV: maximum field of view for cars when they are initialized.
* keepBestCar: if true, the best car of the previous generation will be transferred to the new generation without any genetical changes.

## Training
Start the application by opening a console and type `python main.py`. A window will open and you can draw a training track. The track should be a closed loop. Hit Enter and a population of cars will be generated that try to complete the track you have provided for them.
Some fitness scores will be displayed in the console while training.
IMPORTANT: The best car has to complete at lest 10% of the track, otherwise a new population of cars will be generated randomly.

## Testing
Once you are satisfied with the training results, hit Enter again and you can draw a new track that the trained cars can drive on. This can be used as a test if the learned parameters during training generalize well to a new unknown environment.
After training a text file will be generated that stores information about the best car of each generation. This can be used to visualize the training results.

# Requirements
In order to run this application you will need the following modules:
* PyGame
* NumPy
* skimage
* PIL
* Pandas
* matplotlib
