# human-activity-recognition(HAR)
This repository provides the codes and data used in our paper "The layer-wise training convolutional neural
networks using local loss for sensor based human activity recognition", where we implement and evaluate several state-of-the-art approaches, ranging from handcrafted-based methods to convolutional neural networks. Also, we standardize a large number of datasets, which vary in terms of sampling rate, number of sensors, activities, and subjects.

The layer-wise training convolutional neural networks using local loss for sensor based human activity recognition.
(link: https://ieeexplore.ieee.org/abstract/document/9026890/)

the OPPOTUNITY and PAMAP2 dataset are to large to upload, and the remaining three datasets in this paper can be obtained. If you need OPPOTUNITY and PAMAP2 datasets, please contact us by email (leizhang@njnu.edu.cn or teqi159@gmail.com).

Welcome to cite our paper！(https://ieeexplore.ieee.org/abstract/document/9026890/)

Please cite our paper in your publications if it helps your research.

@article{teng2020layer,
  title={The layer-wise training convolutional neural networks using local loss for sensor based human activity recognition},
  author={Teng, Qi and Wang, Kun and Zhang, Lei and He, Jun},
  journal={IEEE Sensors Journal},
  year={2020},
  publisher={IEEE}
}

# Requirements

● Python3

● PyTorch (Recommended version 1.2.0+cu92)

● Scikit-learn

● Numpy

# Abstract:
Recently, deep learning, which are able to extract automatically features from data, has achieved state-of-the-art performance across a variety of sensor based human activity recognition (HAR) tasks. However, the existing deep neural networks are usually trained with a global loss, and all hidden layer weights have to be always kept in memory before the forward and backward pass has completed. The backward locking phenomenon prevents the reuse of memory, which is a crucial limitation for wearable activity recognition. In the paper, we proposed a layer-wise convolutional neural networks (CNN) with local loss for the use of HAR task. To our knowledge, this paper is the first that uses local loss based CNN for HAR in ubiquitous and wearable computing arena. We performed experiments on five public HAR datasets including UCI HAR dataset, OPPOTUNITY dataset, UniMib-SHAR dataset, PAMAP dataset, and WISDM dataset. The results show that local loss works better than global loss for tested baseline architectures. At no extra cost, the local loss can approach the state-of-the-arts on a variety of HAR datasets, even though the number of parameters was smaller. We believe that the layer-wise CNN with local loss can be used to update the existing deep HAR methods.

# References:
[1] J. Wang, Y. Chen, S. Hao, X. Peng, and L. Hu, “Deep learning for sensorbased activity recognition: A survey,” Pattern Recognition Letters, vol. 119, pp. 3–11, 2019.

[2] P. Rashidi and D. J. Cook, “Keeping the resident in the loop: Adapting the smart home to the user.” IEEE Trans. Systems, Man, and Cybernetics, Part A, vol. 39, no. 5, pp. 949–959, 2009.

[3] Y.-J. Hong, I.-J. Kim, S. C. Ahn, and H.-G. Kim, “Mobile healthmonitoring system based on activity recognition using accelerometer,”
Simulation Modelling Practice and Theory, vol. 18, no. 4, pp. 446–455,2010.

[4] S.-R. Ke, H. L. U. Thuc, Y.-J. Lee, J.-N. Hwang, J.-H. Yoo, and K.-H.Choi, “A review on video-based human activity recognition,” computers,vol. 2, no. 2, pp. 88–131, 2013.

[5] O. D. Lara and M. A. Labrador, “A survey on human activity recognitionusing wearable sensors,” IEEE communications surveys & tutorials,vol. 15, no. 3, pp. 1192–1209, 2012.

[6] L. Chen, J. Hoey, C. D. Nugent, D. J. Cook, and Z. Yu, “Sensorbased activity recognition,” IEEE Transactions on Systems, Man, and
Cybernetics, Part C (Applications and Reviews), vol. 42, no. 6, pp. 790–808, 2012.

[7] D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, “Humanactivity recognition on smartphones using a multiclass hardware-friendly  support vector machine,” in International workshop on ambient assisted
living. Springer, 2012, pp. 216–223.

[8] P. Casale, O. Pujol, and P. Radeva, “Human activity recognition fromaccelerometer data using a wearable device,” in Iberian Conference on Pattern Recognition and Image Analysis. Springer, 2011, pp. 289–296.

[9] M. Berchtold, M. Budde, D. Gordon, H. R. Schmidtke, and M. Beigl, “Actiserv: Activity recognition service for mobile phones,” in International Symposium on Wearable Computers (ISWC) 2010. IEEE, 2010, pp. 1–8.

[10] W. Jiang and Z. Yin, “Human activity recognition using wearable sensorsby deep convolutional neural networks,” in Proceedings of the 23rd ACM international conference on Multimedia. Acm, 2015, pp. 1307–1310.

[11] F. Ordóñez and D. Roggen, “Deep convolutional and lstm recurrent neural networks for multimodal wearable activity recognition,” Sensors,vol. 16, no. 1, p. 115, 2016.

[12] M. Jaderberg, W. M. Czarnecki, S. Osindero, O. Vinyals, A. Graves, D. Silver, and K. Kavukcuoglu, “Decoupled neural interfaces using
synthetic gradients,” in Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017, pp. 1627–1635.

[13] A. N. Gomez, M. Ren, R. Urtasun, and R. B. Grosse, “The reversible residual network: Backpropagation without storing activations,” in Advances in neural information processing systems, 2017, pp. 2214–2224.

[14] Y. Bengio, D.-H. Lee, J. Bornschein, T. Mesnard, and Z. Lin, “Towards biologically plausible deep learning,” arXiv preprint arXiv:1502.04156, 2015.

[15] X. Glorot, A. Bordes, and Y. Bengio, “Deep sparse rectifier neural networks,” in Proceedings of the fourteenth international conference on artificial intelligence and statistics, 2011, pp. 315–323.

[16] A. Nøkland and L. H. Eidnes, “Training neural networks with local error signals,” arXiv preprint arXiv:1901.06656, 2019.

[17] D. Roggen, A. Calatroni, M. Rossi, T. Holleczek, K. Förster, G. Tröster,P. Lukowicz, D. Bannach, G. Pirkl, A. Ferscha et al., “Collecting complex activity datasets in highly rich networked sensor environments,” in 2010 Seventh international conference on networked sensing systems (INSS). IEEE, 2010, pp. 233–240.

[18] D. Micucci, M. Mobilio, and P. Napoletano, “Unimib shar: A dataset for human activity recognition using acceleration data from smartphones,” Applied Sciences, vol. 7, no. 10, p. 1101, 2017.

[19] A. Reiss and D. Stricker, “Introducing a new benchmarked dataset for activity monitoring,” in 2012 16th International Symposium on Wearable Computers. IEEE, 2012, pp. 108–109.

[20] D. Ravi, C. Wong, B. Lo, and G.-Z. Yang, “Deep learning for human activity recognition: A resource efficient implementation on low-power devices,” in 2016 IEEE 13th International Conference on Wearable and Implantable Body Sensor Networks (BSN). IEEE, 2016, pp. 71–76.
[21] M. Janidarmian, A. Roshan Fekr, K. Radecka, and Z. Zilic, “A comprehensive analysis on wearable acceleration sensors in human activity recognition,” Sensors, vol. 17, no. 3, p. 529, 2017.

[22] T. Plötz, N. Y. Hammerla, and P. L. Olivier, “Feature learning for activity recognition in ubiquitous computing,” in Twenty-Second International Joint Conference on Artificial Intelligence, 2011.

[23] L. Bao and S. S. Intille, “Activity recognition from user-annotated acceleration data,” in International conference on pervasive computing. Springer, 2004, pp. 1–17.

[24] T. Huynh and B. Schiele, “Analyzing features for activity recognition,” in Proceedings of the 2005 joint conference on Smart objects and ambient intelligence: innovative context-aware services: usages and technologies. ACM, 2005, pp. 159–163.

[25] M. Zeng, L. T. Nguyen, B. Yu, O. J. Mengshoel, J. Zhu, P. Wu, and J. Zhang, “Convolutional neural networks for human activity recognition using mobile sensors,” in 6th International Conference on Mobile Computing, Applications and Services. IEEE, 2014, pp. 197–205.

[26] K. Wang, J. He, and L. Zhang, “Attention-based convolutional neural network for weakly labeled human activities recognition with wearable sensors,” IEEE Sensors Journal, 2019.
[27] D. Kuang, C. Ding, and H. Park, “Symmetric nonnegative matrix factorization for graph clustering,” in Proceedings of the 2012 SIAM
international conference on data mining. SIAM, 2012, pp. 106–117.

[28] O. Banos, J.-M. Galvez, M. Damas, H. Pomares, and I. Rojas, “Windowsize impact in human activity recognition,” Sensors, vol. 14, no. 4, pp.6474–6499, 2014.

[29] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin,A. Desmaison, L. Antiga, and A. Lerer, “Automatic differentiation in pytorch,” 2017.

[30] C. A. Ronao and S.-B. Cho, “Human activity recognition with smartphone sensors using deep learning neural networks,” Expert systems with applications, vol. 59, pp. 235–244, 2016.

[31] A. Ignatov, “Real-time human activity recognition from accelerometer data using convolutional neural networks,” Applied Soft Computing, vol. 62, pp. 915–922, 2018.

[32] L. Zhang, X. Wu, and D. Luo, “Recognizing human activities from raw accelerometer data using deep neural networks,” in 2015 IEEE
14th International Conference on Machine Learning and Applications (ICMLA). IEEE, 2015, pp. 865–870.

[33] N. Y. Hammerla, S. Halloran, and T. Plötz, “Deep, convolutional, and recurrent models for human activity recognition using wearables,” arXiv preprint arXiv:1604.08880, 2016.

[34] Z. Yang, O. I. Raymond, C. Zhang, Y. Wan, and J. Long, “Dfternet: towards 2-bit dynamic fusion networks for accurate human activity
recognition,” IEEE Access, vol. 6, pp. 56 750–56 764, 2018.

[35] F. Li, K. Shirahama, M. Nisar, L. Köping, and M. Grzegorzek, “Comparison of feature learning methods for human activity recognition using wearable sensors,” Sensors, vol. 18, no. 2, p. 679, 2018.

[36] A. Khan, N. Hammerla, S. Mellor, and T. Plötz, “Optimising sampling rates for accelerometer-based human activity recognition,” Pattern

Recognition Letters, vol. 73, pp. 33–40, 2016.
[37] M. Zeng, H. Gao, T. Yu, O. J. Mengshoel, H. Langseth, I. Lane, and X. Liu, “Understanding and improving recurrent networks for human
activity recognition by continuous attention,” in Proceedings of the 2018 ACM International Symposium on Wearable Computers. ACM, 2018,
pp. 56–63.

[38] M. A. Alsheikh, A. Selim, D. Niyato, L. Doyle, S. Lin, and H.-P. Tan, “Deep activity recognition models with triaxial accelerometers,” in Workshops at the Thirtieth AAAI Conference on Artificial Intelligence, 2016.

[39] J.-L. Reyes-Ortiz, L. Oneto, A. Samà, X. Parra, and D. Anguita, “Transition-aware human activity recognition using smartphones,” Neurocomputing, vol. 171, pp. 754–767, 2016.

[40] Girish,”Human activity recognition using Recurrent NeuralNets RNN LSTM and Tensorflow on Smartphones”, https://github.com/girishp92/Human-activity-recognition-usingRecurrent-Neural-Nets-RNN-LSTM-and-Tensorflow-on-Smartphones.
