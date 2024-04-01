## ECE 766
### Homework 6

### Challenge 1a
This challenge was really tricky. To start, I used a for loop for just the index_map and then worked out the logic. However, that didn't work and I wound up just vectorizing it instead anyway. 

To start, for my focus measure, I used a list comporehension to convolve the laplacian kernel accross all the images in `gray_list`. After that, I squared them, and then summed them up again along the same window with a convolution of a kernel of ones. For my final convolution, I used an averaging kernel based on the window size.

After I was done convolved, I used `np.stack()` to change my list of images into a 3D array. This way, instead of looping again, I could use the built in numpy functionality. Finally, to get the indicies, I just `np.argmax()` accross the entire 3D array.

The image `index_map.png` is scaled up to be more clearly displayed and understood. The picture that is actually used in part b is `index.png`.

### Challenge 1b
This challenge was relatively straightforward. All I do is register the user click (provided it's valid) and display the image from the index found at the `index_map` at that point.