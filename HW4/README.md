## CS 766 - Homework 4
---
### Challenge 1a
This part of the challenge largely had me relying on the slides and videos for most of it. Fairly straightforward.

### Challenge 1b
I used the information from the slides here as well. I did run into an issue where my image was flipped, but that was because I was forgeting to change my reference from the cartesian coordinates used in the slides to the standard image coordinate from (i.e. moving the origin from the bottom left to the top left)

### Challenge 1c
runRANSAC took a hot minute to complete, but the code wasn't actually too complex once I figured it out. However, once we got it working, we found that our results were inconsistent due to using the `np.random.choice` function. To circumvent this and get more consistent results, I chose to also set a seed in the runRANSAC function to consitently get good results based on my number of iterations and epsilon.

### Challenge 1d
I largely relied on the `ndimage.distance_transform_edt` used in the notes to generate the weight mask. But then I blended it to preserve the edges. If I didn't do that, I was left with a very dull image around the entire boundary, not just the boundary where the images met.

### Challenge 1e
This part was the most satisfying. Everything finally all came together. There was a padding issue that made it so the image "grew" to the right and would shrink as it image got warped, so Keshav and I worked on a solution to dynamically calculate the blank canvas shape to avoid that.

### Challenge 1f
This one should have fallen right into line with 1e, but it unfortunately didn't. My first few passes crashed my computer. However, I looked into it and there were three main issues: 
- I used my native camera, so the size was in the MB range and not KBs.
- The resolution was higher than it needed to be.
- I was taking pictures too closely together, so even the slightest angle shift resulted in too great of a shift in perspective.
To resolve all these issues, I wound up taking three more pictures using SnapChat, since it produces a way lower quality image. The resulting images were around the 250KB range. Using these new images I was able to create the panorama in about 2 minutes of runtime.