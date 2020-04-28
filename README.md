# human-activity-recognition

The layer-wise training convolutional neural networks using local loss for sensor based human activity recognition.
(link: https://ieeexplore.ieee.org/abstract/document/9026890/)

the OPPOTUNITY and PAMAP2 dataset are to large to upload, and the remaining three datasets in this paper can be obtained. If you need OPPOTUNITY and PAMAP2 datasets, please contact us by email (leizhang@njnu.edu.cn or teqi159@gmail.com).

Welcome to cite our paper！(https://ieeexplore.ieee.org/abstract/document/9026890/)

# Abstract
Recently, deep learning, which are able to extract automatically features from data, has achieved state-of-the-art performance across a variety of sensor based human activity recognition (HAR) tasks. However, the existing deep neural networks are usually trained with a global loss, and all hidden layer weights have to be always kept in memory before the forward and backward pass has completed. The backward locking phenomenon prevents the reuse of memory, which is a crucial limitation for wearable activity recognition. In the paper, we proposed a layer-wise convolutional neural networks (CNN) with local loss for the use of HAR task. To our knowledge, this paper is the first that uses local loss based CNN for HAR in ubiquitous and wearable computing arena. We performed experiments on five public HAR datasets including UCI HAR dataset, OPPOTUNITY dataset, UniMib-SHAR dataset, PAMAP dataset, and WISDM dataset. The results show that local loss works better than global loss for tested baseline architectures. At no extra cost, the local loss can approach the state-of-the-arts on a variety of HAR datasets, even though the number of parameters was smaller. We believe that the layer-wise CNN with local loss can be used to update the existing deep HAR methods.
