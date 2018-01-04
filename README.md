This neural system for visual question answering is roughly based on the paper "Dynamic Memory Networks for Visual and Textual Question Answering" by Xiong et al. (ICML2016). It is implemented using the Tensorflow library, and allows end-to-end training of both CNN and RNN parts. To use it, you will need the Tensorflow version of VGG16 or ResNet 50/101/152, which can be obtained with Caffe-to-Tensorflow. 

**The code is now compatible with Tensorflow r1.4**.

References
----------

* [Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417) Caiming Xiong, Stephen Merity, Richard Socher. ICML 2016.
* [Visual Question Answering (VQA) dataset](http://visualqa.org/)
* [Implementing Dynamic memory networks by YerevaNN](https://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/)
* [Dynamic memory networks in Theano](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano)
* [Dynamic Memory Networks in Tensorflow](https://github.com/therne/dmn-tensorflow)
* [Caffe to Tensorflow](https://github.com/ethereon/caffe-tensorflow)

