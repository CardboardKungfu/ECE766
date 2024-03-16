## Computer Vision - Homework 5

### Challenge1a
Fairly straightforward. I used `skimage` functions to essentially find the largest contour (e.g. the circle), and then took the mean to find the centroid. From there it was just a matter of finding the area and then the radius.

### Challenge1b
This was probably the hardest part of the homework. But it turns out that the formula was fairly simple. I centered my circle on the origin by subtracting off the center. From there, I treated our image as the reflectence map and then calculated z based on the euclidian distance from the origin. This worked well since we are centered around the origin. After that, it was a simple process of normalizing our vector and then scaling by the brightest pixel in our image.

> Assume that this is the direction of the corresponding light source (Why is it safe to assume this?)

Since the Lambertian Model is a simplified version of a BRDF, then we can use the Helmholtz Reciprocity to see that swapping the light source and the collection point doesn't change the normal or the brightness. Additionally, since we're dealing with an isotropic Lambertian Model (or rather a 3D BRDF), then we know the surface is rotationally symmetric.

### Challenge1c
For this part, it only took two lines. I stacked all the images on top of each other into a 3D numpy array and then summed them along the 3rd axis. Since the foreground is defined as any point where a 1 shows up along any of the images, summing them along the 3rd axis finds that without any complicated flow control.

### Challenge1d
Here I looped through the images and only considered points where the mask was one. From there, I just used the formulas in the notes to find the albedos and normals. I chose to use the psuedo-inverse accross the board for robustness. This controls for non-square light sources, but also for when our light source lies in a plane.