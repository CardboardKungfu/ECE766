ECE 766
Jed Pulley
HW2

Walkthrough:
This was fairly straightforward. All I did was choose a structuring element of size 3 and then followed
the steps laid out. For removing the rice, I chose a structuring element of size 25 and that cleaned them right up

Challenge 1A:
This function only required a few lines of code. I followed the example from the walkthrough and demoTricksFun
to threshold the image based on the given threshold. By using the '>' operator, I was able to create a numpy
array of bools, essentially. And since true and false can be treated as 1 and 0 respectively, it made my life
fairly easy

Challenge 1B:
This where things got a bit trickier. I worked with Hemanth and Keshav for help on this, but the code and flow
is my own. I followed the slides and found the necessary arguments needed to create the database. I had a few
minor syntax errors that provided the bulk of my debugging time, but largely all the formulas worked according
to how they were shown in the slides. I also added a section that reads in two separate databases for Challenge 1C here, which I elaborated on below.

Challenge 1C:
I chose to use only the minimum moment of inertia and roundness for my object detection criteria. The reason
I went with this is because objects can be oriented in any direction, their centers can change, and we can't
be certain that their area will remain the same (granted, it was given in Piazza that the size doesn't change). With these two parameters chosen, I mostly just trial and errored/spitballed my threshold percentages. From there it was only a matter of comparing the respective objects.

Challenge 1C - In Addition:
To reduce run time, I coded in the 'in addition' section into 1B. I generated both databases and stored them
as .npy files. That way, when 1C is called, it can be run with both databases and will show the detected objects.
Some of my results don't look 100% accurate, but I also have a hard time arguing with them as well. It's hard
to eyeball what the correct orientation should be in regard to moment of inertia. Regardless, the only two
objects common to all images was the floppy disk looking thing and the triangular clip thing. Both databases
held those objects and consequently displayed them. The image that pops up that has all the objects annotated with centers and orientations is where I loaded it into the separate database, similar to part 1B.