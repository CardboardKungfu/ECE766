Homework 1
Jed Pulley

Walkthrough2 Process:
-For creating the green and blue images, I basically just followed the given template. 
-To create the collage, I recognized that the images were just np.arrays which can be stacked. So first I used np.hstack to combine the left 
and right sides, and then used vstack to stack them vertically on top of each other

Walkthrough3 Process:
- For my threshold value, I just played around until I was happy. No particular reason.
- For choosing my blue and green channel values, at first I considered setting the pixel to 0 wherever a red value was > 0.
However, I realized that we already had a perfectly resized mask in the variable iresized_mask. Since this was a white mask, I used the '~' bit flip operator 
to get a black mask and then just multiplied it by both color channels, thus only whiping out the values I didn't want