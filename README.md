# trishul
The goal of this project is to create a deep learning model that can map user input sEMG signals to kinematic outputs for proshtetics. 

The dataset we used can be found at https://ninapro.hevs.ch/instructions/DB1.html we are using Subject 01's data

We are given sEMG data as our input through 10 columns of sEMG data gathered through electrodes .

Columns 1-8 are the electrodes equally spaced around the forearm at the height of the radio humeral joint. Columns 9 and 10 contain signals from the main activity spot of the muscles flexor and extensor digitorum superficialis. 

The electrodes are 10 Otto Bock MyoBock 13E200 electrodes

Our output is kinematic data given through 22 sensors on the Cyberglove 2 data glove.

The dataset includes 10 repetitions of 52 different movements. The subjects are asked to repeat movements which were shown as movies on the screen of a laptop.
The experiment is divided in three exercises:

Basic movements of the fingers.
Isometric, isotonic hand configurations and basic wrist movements.
Grasping and functional movements.

The data is mapped such that there is a row which represents a specific time at which the data was collected. There are 10 columns of electrode data which contain the voltages collected by that electrode at that time. At the same time, there is a row of data collected by the kinematic glove's 22 sensors which represent the output caused by the electrodes at the same time stamp.
