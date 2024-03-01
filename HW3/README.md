# Walkthrough 1
Fairly straightforward. Mostly just plug and play.

# Challenge 1A
Here I chose to use Canny edge detection. I tried sobel as well, but it didn't perform nearly as well. I chose sigma and my thresholds based on trial and error.

# Challenge 1B
Keshav Sharan and I worked on this for a LONG time. We initially were just doing simple voting and thresholding to generate our accumulator, but it was missing peaks that fell below decent thresholds.
After MANY different methods, we decided on a method that found regional peaks within our already created
accumulator array. It scanned over with a window and then we created a new accumulator array with the regional
peaks set very bright. In conjunction with this, we both utilizated a soft voting technique based on a 3x3
kernel with a peak in the center and lower values on the perimeter.

# Challenge 1C
This part was more of just fine tuning. The biggest hurdle was getting the results from 1B properly. Once
we got those squared away, it was just a matter of finding the right thresholds. The most challenging aspect
was that once you changed a variable all the way back in 1A, it cascaded down the entire chain and you had to
retest everything. It took quite a few iterations, but we got there in the end.

# Challenge 1D
I really struggled to come up with a good strategy for clipping the lines. The best solution I found was to
just use our edge image as a mask. I created a blank background to also draw the hough lines on, masked it with a dilated edge image, and then pasted that ontop of the original image. This method worked best on the first and second images, but the third image was far more fussy since the vase had a bunch of squiggly lines. After a few too many hours of tweaking, I finally landed on what I thought was the best achievable result.