# Eyetracking-Data-analysis-and-scanpath-comparison
This code was based in academic research in an attempt to evaluate the visual routine similarities of different observers during an eyetracking experiment. I am not a specialist and this work was developed as my bachalor's thesis. Unfortunately i don't have the dataset anymore, but you can see the data disposition at (https://www.realeye.io/) after any experiment.

The eyetraking data was obtained from a Webcam-based eyetraking system (https://www.realeye.io/), so the structure can possibly be ineficient without the proper modifications to work with other eyetraking systems.

Since the data files were on .CSV, the first step was reading these files and pre-processing all the informations contained: x-coordinates, y-coordinates, timestamps, etc.

All the observer's data were arregend into matrixes to offer a better management.

To compare visual routines, is necessary to stipulate parameters, in this case, i was looking for two special eye movements: fixations (related to visual atention) and saccades.

It is possible to find eyetrackers that already proccess the visual data and return the fixations, but since the eyetracker in this project was based on webcams, i have chosen to use algorithms to detect fixations by myself and filter these fixations by grouping them. All of this happened after the proper bibliographyc investigation and the references will be listed at the end of this README.md section.

Fixations were detected using a Velocity Threshold algorithym. Therefore was necessary to calculate the velocity of the eye movements based in angle variation of the center of pupil (using the x and y coordinates at the observed image) and velocity threshold. 

After detecting the fixations was necessary to filter them. This was important to avoid spliting fixations that are in reality the same one. This process was based on the durations and on the maximum angle variation of each fixation according to what is described in literature.

The scanpaths comparison to obtain similarity indexes has occurred with two techniques: Edit Distance and ScanMatch. Both required a transformation of the visual routines into strings (where each fixation is represented by a character). This transformation is based in image segmentation and to this code the segmentation was mabe by spliting the images areas into 25 equal regions with the same dimensions: each region was associated to a random and unique letter.

Edit Distance compares the strings based on Levenshtein Distance, and ScanMatch uses a statistical approach with a score matrix (BLOSUM matrix in my case).

The **main.py** is commented and was summarized into this README.md in an attempt to give you as much information as possible about the parameters and methods, but may be necessary do further investigation for yourself to understand some of the main techniques, as the BLOSUM matrixes incorporated into the ScanMatch approach.

Some functions demands a little bit more explanation:

- **expandir(string,duracao)** : responsable to return a string after multiplicate the fixations according to a duration threshold (50 milliseconds in this code).
- **blosum_compare(string1,string2)** : compare two strings and returns a similarity score based on ScanMatch.
- **score_levenshtein(string1,string2)** : compare two strings and returns a similarity score based on Edit Distance.
- **string_expandida2(string_expandida)** : expands the strings to avoid blank gaps between fixations, turning those gaps into **Z** letters (the goal was not punish gaps).

This code also offers saliency maps as an option and it is quite simple modify that to plot the scanpaths into the observed images.
May be useful to you applying some changes as compare the fixation durations at every image's regions and plot a graph based on that (shame on me i DID that but i was versioning the code by myself and have lost some features that right now i don't have free time do re-do).

REFERENCES : WELL, COULD BE A LOT OF WORK BRINGING THERE ALL THE ACADEMIC REFERENCES SO I'LL PUT MY BACHELOR'S THESIS AT THIS REPOSITORY (IT IS IN PORTUGUESE BUT YOU CAN FIND ALL REFERENCES CLOSE TO THE END OF THE PDF FILE SO GOOD LUCK)





