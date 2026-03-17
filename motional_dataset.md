The nuPlan v1.1 dataset brings multiple improvements over the v1.0 dataset - scenario tagging frequency has been significantly increased, scenario tagging performance has been improved, traffic light labels and planned routes are now more accurate. For information on how to get started, see the devkit at github.com/motional/nuplan-devkit.

nuPlan Test Set
minus
Test Sensors for nuPlan.

nuPlan Train Set
minus
Train Sensors for nuPlan.

nuPlan Val Sensors
minus
Val Sensors for nuPlan.

nuPlan Maps
minus
nuPlan maps, required for the full dataset.

download
Zip containing Maps  
[US, Asia]
0.90 GB
nuPlan Mini Split
minus
Mini split for nuPlan.

download
Zip containing log databases for the mini split  
[US, Asia]
7.96 GB
nuPlan Mini Sensors
minus
Mini sensors for nuPlan.

nuPlan Train Split
minus
Training split for nuPlan.

download
Zip containing log databases for the training split for Boston City  
[US, Asia]
35.54 GB
download
Zip containing log databases for the training split for Pittsburgh City  
[US, Asia]
28.52 GB
download
Zip containing log databases for the training split for Singapore City  
[US, Asia]
32.56 GB
plus
download
Zip containing log databases for the training split for Las Vegas City  
850.80 GB
nuPlan Val Split
minus
Validation split for nuPlan.

download
Zip containing log databases for the validation split  
[US, Asia]
90.30 GB
nuPlan Test Split
minus
Test split for nuPlan - note that this is different from the hidden test set that is used in the nuPlan competition.

download
Zip containing log databases for the test split  
[US, Asia]
89.33 GB
nuPlan v1.0 Datasetexpand

The nuPlan v1.0 dataset consists of over 15,000 logs and 1300+ hours of driving data. The data is recorded over 4 cities - Boston, Pittsburgh, Singapore and Las Vegas. Due to the huge size of nuPlan, the mini, val, train and test splits are available for download separately. The mini split with 72 logs allows the user to explore the database without having to download the complete dataset. It will also be possible to download raw sensor data (camera and lidar) for around 10% of the logs. However, this data will be made available later. For information on how to get started, see the devkit at github.com/motional/nuplan-devkit.

nuPlan Maps
minus
nuPlan maps, required for full dataset

download
Zip containing Maps  
[US, Asia]
0.90 GB
nuPlan Mini Split
minus
Mini split for nuPlan.

download
Zip containing Mini DBs  
[US, Asia]
8.25 GB
nuPlan Val Split
minus
Val split for nuPlan.

download
Zip containing Val DBs  
[US, Asia]
93.65 GB
nuPlan Test Split
minus
Test split for nuPlan.

download
Zip containing Test DBs  
[US, Asia]
89.24 GB
nuPlan Train Split
minus
Train split for nuPlan.

download
Zip containing Train DBs for Boston City  
[US, Asia]
37.80 GB
download
Zip containing Train DBs for Pittsburgh City  
[US, Asia]
29.68 GB
download
Zip containing Train DBs for Singapore City  
[US, Asia]
37.80 GB
plus
download
Zip containing Train DBs for Las Vegas City  
180.33 GB
Fact Sheet - nuPlan Dataset v1.0

Location	Total	Las Vegas	Singapore	Pittsburgh	Boston
Duration	
1312h
845h
(64%)
196h
(15%)
155h
(12%)
115h
(9%)
Logs	16733	9584	1945	2007	3197
Scenarios	
● Total unique scenario types: 75
● Scenario categories
● Dynamics: 5 types (e.g. high lateral acceleration)
● Interaction: 18 types (e.g. waiting for pedestrians to cross)
● Zone: 8 types (e.g. on pickup-dropoff area)
● Maneuver: 22 types (e.g. unprotected cross turn)
● Behavior: 22 types (e.g. stopping at a traffic light with a lead vehicle ahead)
Tracks	
● Frequency of tracks/ego: 20hz
● Number of unique tracks: 38,831,238
● Average length of tracks: 9.64s
Data samples	
● Total data points (lidarpcs): ~95 million
● Total scenarios tagged: ~3 million
Object classes	
Vehicle, Bicycle, Pedestrian, Traffic cone, Barrier, Construction zone sign, Generic object
Other information available	
● Route information
● Goal poses
● A sequence of roadblock ids toward the goal
● Traffic light information
● Lane connector id associated with traffic light
● Traffic light status (green, red, unknown)
