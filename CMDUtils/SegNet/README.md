**SegNet** is an utility, based on tiny-dnn and Opencv, for SegNet-like architecture CNN training and testing.

Comand line utility for the image segmentation by means of CNN

*To train CNN use:*

-i[dirname]  - directory with training data

-o[filename] - name for the output file with trained network

-e[int] - number of training epoch (ref. to tiny-dnn doc.)

-m[int] - minibatch size for training (ref. to tiny-dnn doc.)

-c[int] - to what cols training images should be resized

-r[int] - to what rows training images should be resized

-x[int] - desired number of input channels (0 means same as in source)

-y[int] - desired number of output channels (0 means same as in source)

*To segment image by CNN:*

-n[filename] - name for the file with pretrained network

-s[filename] - image for segmentation

-a[filename] - where to save segmented image

Alex A. Taranov, based on Qt, Opencv and tiny-dnn" 
