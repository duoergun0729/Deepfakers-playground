# Papers, tools and data that used in Generating Deepfakes

For the convenience of reading and searching, the following is a list of papers about Deepfakes generation. The content might  be updated from time to time.

***Content***

Vedio & Images:

- Korshunova et al. **[Fast Face-swap Using Convolutional Neural Networks](http://openaccess.thecvf.com/content_iccv_2017/html/Korshunova_Fast_Face-Swap_Using_ICCV_2017_paper.html)**
- Razavi et al. **[Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446)**
- Turkoglu et al. **[A Layer-Based Sequential Framework for Scene Generation with GANs](https://arxiv.org/abs/1902.00671)**
  - Code:(https://github.com/0zgur0/Seq_Scene_Gen)
- Denton et al. **[Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](https://arxiv.org/abs/1506.05751)**
- Gatys et al. **[Image style transfer using convolutional neural networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)**
- Goodfellow et al. **[Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)**
- Chan et al. **[Everybody Dance Now](https://arxiv.org/pdf/1808.07371.pdf)**
- Karras et al. **[A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)**
- Kim et al. **[Deep video portraits](https://web.stanford.edu/~zollhoef/papers/SG2018_DeepVideo/page.html)**
- Vondrick et al. **[Generating Videos with Scene Dynamics](http://www.cs.columbia.edu/~vondrick/tinyvideo/paper.pdf)**
- Klawonn et al. **[Generating Triples with Adversarial Networks for Scene Graph Construction](https://arxiv.org/pdf/1802.02598.pdf)**
- Thies et al. **[Face2Face: Real-time Face Capture and Reenactment of RGB Videos](http://openaccess.thecvf.com/content_cvpr_2016/papers/Thies_Face2Face_Real-Time_Face_CVPR_2016_paper.pdf)**
- Shen et al. **[“Deep Fakes” using Generative Adversarial Networks (GAN)](http://noiselab.ucsd.edu/ECE228_2018/Reports/Report16.pdf)**
- Wang et al. **[Generative Adversarial Networks: A Survey and
Taxonomy](https://arxiv.org/pdf/1906.01529.pdf)**

Dialogue Response:

- Wu et al. **[Are You Talking to Me? Reasoned Visual Dialog Generation through Adversarial Learning](https://arxiv.org/abs/1711.07613)**
- Olabiyi et al. **[Multi-turn Dialogue Response Generation in an Adversarial Learning Framework](https://arxiv.org/abs/1805.11752)**

Audio:

- Chandna et al. **[WGANSing: A Multi-Voice Singing Voice Synthesizer Based on the Wasserstein-GAN](https://arxiv.org/pdf/1903.10729.pdf)**
- Oord et al. **[WAVENET: A GENERATIVE MODEL FOR RAW AUDIO](https://arxiv.org/pdf/1609.03499.pdf)**
- Vougioukas et al. **[Video-Driven Speech Reconstruction using Generative Adversarial
Networks](https://arxiv.org/pdf/1906.06301.pdf)**

Clothes:
- Kubo et al. **[GENERATIVE ADVERSARIAL NETWORK-BASED VIRTUAL TRY-ON WITH CLOTHING REGION](https://openreview.net/pdf?id=B1WLiDJvM)**



***In this section, we made some brief summary of some papers:***

Vedio & Images:

- Korshunova et al. **[Fast Face-swap Using Convolutional Neural Networks](http://openaccess.thecvf.com/content_iccv_2017/html/Korshunova_Fast_Face-Swap_Using_ICCV_2017_paper.html)**: This paper comes up with a solution to face swapping problem in terms of style transfer using CNN.The proposed system in this paper has two additional components performing face alignment and background/hair/skin segmentation.  Facial keypoints were extracted using dlib, which are used to align a frontal view reference face.  Segmentation is used to re- store the background and hair of the input image X based on the cloning technique in OpenCV.Its transformation network here is a multiscale architecture with branches operating on different downsampled versions of the input image X, which is based on the architecture of Ulyanov et al.  Each such branch has blocks of zero-padded convo- lutions followed by linear rectification. Branches are combined via nearest-neighbor upsampling by a factor of two and concatenation along the channel axis. The last branch of the network ends with a 1 × 1 convolution and 3 color channels. For every input image X, it aims to generate an X which jointly minimizes the following content and style loss. These losses are defined in the feature space of the normalised version of the 19-layer VGG network. Besides, this paper applies light loss to solve the problem that the lighting conditions of the content image x are not preserved in the generated image.

- Razavi et al. **[Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446)**: This paper is based on Vector Quantized Variational AutoEncoder and succeed to scale and enhance its performance. The proposed models combines VQ-VAE and PixelCNN, of which the proposed method follows a two-stage approach: first, it trains a hierarchical VQ-VAE to encode images onto a discrete latent space, and then it fits a powerful PixelCNN prior over the discrete latent space induced by all the data. As opposed to vanilla VQ-VAE, in this work it uses a hierarchy of vector quantized codes to model large images. The main motivation behind this is to model local information, such as texture, separately from global information such as shape and geometry of objects. The prior model over each level can thus be tailored to capture the specific correlations that exist in that level. In order to further compress the image, and to be able to sample from the model learned during stage 1, it learns a prior over the latent codes. Fitting prior distributions using neural networks from training data has become common practice, as it can significantly improve the performance of latent variable models. The fidelity of  best class conditional samples are competitive with the state of the art Generative Adversarial Networks, with broader diversity in several classes, contrasting our method against the known limitations of GANs. 

- Turkoglu et al. **[A Layer-Based Sequential Framework for Scene Generation with GANs](https://arxiv.org/abs/1902.00671)**: The paper tackles the scene composition task with a layered structure coding approach, which helps reduce the complexity of the problem with one subproblem at a time. The idea is based on process of landscape painting from overall structure, e.g. mountain ranges, to other individual detailed objects, e..g. animals or trees. The main objective is to compose a realistic scene with allowing element-level control. The generator model is broken into two simpler subtasks. The first one is a background generator. Given a semantic layout map, the input takes a noise vector z_0 and generates a background image x_0. Loss function is defined similar to typical GAN architecture, but only pixels outside of the semantic layout mask are penalized.  
The second step is to add foreground objects. It takes as input previously generated scene x_(t-1), the current foreground object mask M_t and a noise vector z_t. Conditioned on the previous scene without the regions M_t ignoring, this task is turned into an image pinpointing problem. Loss function takes both local reconstruction loss, i.e. object region and global reconstruction, i.e. whole image. 

- Denton et al. **[Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](https://arxiv.org/abs/1506.05751)**: The papers proposes an approach to break generation problem into a sequence of more manageable stages. At each scale, a CNN-based generative model is implemented. Samples are drawn in a coarse-to-fine fashion, starting with a low-frequency residual image. The subsequent levels sample the band-pass structure at the next level, conditioned on the output from previous scale, until final level. Laplacian pyramid is implemented within the architecture, which is a linear invertible image representation consisting of a set of band-pass images, plus a low-frequency residual. The paper names its network construction as Laplacian Generative Adversarial Networks (LAPGAN). It combines conditional GAN model with Laplacian pyramid representation.

- Gatys et al. **[Image style transfer using convolutional neural networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)**: Rendering the semantic content of an image in different styles is a difficult image processing task. Arguably, a major limiting factor for previous approaches has been the lack of image representations that explicitly represent semantic information and, thus, allow to separate image content from style. Here this paper uses image representations derived from Convolutional Neural Networks optimized for object recognition, which make high level image information explicit. This paper introduces A Neural Algorithm of Artistic Style that can separate and recombine the image content and style of natural images. The algorithm allows us to produce new images of high perceptual quality that combine the content of an arbitrary photograph with the appearance of numerous well-known artworks. The results provide new insights into the deep image representations learned by Convolutional Neural Networks and demonstrate their potential for high level image synthesis and manipulation.

- Goodfellow et al. **[Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)**:Ian proposes a new framework for estimating generative models via an adversarial process, in which two models are simultanously trained: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to ½ everywhere. In the case where G and D are defined by multiplayer perceptrons, the entire system can be trained with backpropagation.  
The models go back and forth many times until the artificial image is practically identical to the original.

- Chan et al. **[Everybody Dance Now](https://arxiv.org/pdf/1808.07371.pdf)**: The paper shows how motions in a source video can be transferred to target people in another video. The method divides into pose detection, global pose normalization, and mapping from normalized pose stick figures to the target subject. In the pose detection stage, we use a pre-trained state-of-the-art pose detector to create pose stick figures given frames from the source video. The global pose normalization stage accounts for differences between the source and target body shapes and locations within the frame. Finally, a system is designed to learn the mapping from the pose stick figures to images of the target person using adversarial training. 

- Karras et al. **[A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)**:Using source images, style-based generator identifies styles such as pose and facial features to produce an image of a fake person. Real people controlling the generator can then change the styles to adjust how the fake person looks.  
![image](https://github.com/11-785-Deepfakers/Deepfakers-playground/raw/master/images/graph3.png)  

- Kim et al. **[Deep video portraits](https://web.stanford.edu/~zollhoef/papers/SG2018_DeepVideo/page.html)**:
The paper transfers the full 3D head position, head rotation, face expression, eye gaze, and eye blinking from a source actor to a portrait video of a target actor. The core of the approach is a generative neural network with a novel space-time architecture. The network takes as input synthetic renderings of a parametric face model, based on which it predicts photo-realistic video frames for a given target actor. The paper renders a synthetic target video with the reconstructed head animation parameters from a source video, and feed it into the trained network – thus enabling source-to-target video re-animation. 

Dialogue Response:

- Wu et al. **[Are You Talking to Me? Reasoned Visual Dialog Generation through Adversarial Learning](https://arxiv.org/abs/1711.07613)**: This paper comes up with a solution for visual dialog generation. The key challenge in visual dialog is thus maintaining a consistent, and natural dialogue while continuing to answer questions correctly. This paper solve this problem with a novel approach that combines Reinforcement Learning and Generative Adversarial Networks (GANs) to generate more human-like responses to questions. The GAN helps overcome the relative paucity of training data, and the tendency of the typical MLE-based approach to generate overly terse answers.  
The model is composed of two components, the ﬁrst being a sequential co-attention generator that accepts as input image, question and dialog history tuples, and uses the co-attention encoder to jointly reason over them. The second component is a discriminator tasked with labelling whether each answer has been generated by a human or the generative model by considering the attention weights. The output from the discriminator is used as a reward to push the generator to generate responses that are indistinguishable from those a human might generate.

- Olabiyi et al. **[Multi-turn Dialogue Response Generation in an Adversarial Learning Framework](https://arxiv.org/abs/1805.11752)**: This paper proposes an adversarial learning approach for generating multi-turn dialogue responses. The GAN's generator is a modified hierarchical recurrent encoder-decoder network (HRED) and the discriminator is a word-level bidirectional RNN that shares context and word embeddings with the generator.  
![image](https://github.com/11-785-Deepfakers/Deepfakers-playground/raw/master/images/graph1.png)  
Left: The hredGAN architecture - The generator makes predictions conditioned on the dialogue history, hi, attention, aji, noise sample, zji, and ground truth, xj−1 i+1. Right: RNN-based discriminator that discriminates bidirectionally at the word level.  
![image](https://github.com/11-785-Deepfakers/Deepfakers-playground/raw/master/images/graph2.png)   
The HRED generator with local attention-The attention RNN ensures local relevance while the context RNN ensures global relevance. Their states are combined to initialize the decoder RNN and the discriminator BiRNN.


# Papers and software in detecting DeepFakes

Here are some papers we found in detecting DeepFakes, other group from defenders' side could add their work below.

***Content***

Dataset:
- **[FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/pdf/1901.08971.pdf)**

Detecting Method:
- **[Xception](https://arxiv.org/pdf/1901.08971.pdf)**

- **[MesoNet: a Compact Facial Video Forgery Detection Network](https://arxiv.org/pdf/1809.00888.pdf)**

- **[A Deep Learning Approach to Universal Image Manipulation Detection Using a New Convolutional Layer](https://dl.acm.org/citation.cfm?id=2930786)**

- **[Distinguishing Computer Graphics from Natural Images Using Convolution Neural Networks](http://www-igm.univ-mlv.fr/~vnozick/publications/Rahmouni_WIFS_2017/Rahmouni_WIFS_2017.pdf)**

- **[Recasting Residual-based Local Descriptors as Convolutional Neural Networks: an Application to Image Forgery Detection](https://arxiv.org/pdf/1703.04615.pdf)**

- **[Rich Models for Steganalysis of Digital Images](https://ieeexplore.ieee.org/document/6197267)**

- **[Deep Learning for Deepfakes Creation and Detection](https://arxiv.org/abs/1909.11573)**

- **[Hybrid LSTM and Encoder–Decoder Architecture for Detection of Image Forgeries](https://ieeexplore.ieee.org/document/8626149)**

- **[Recurrent Convolutional Strategies for Face Manipulation Detection in Videos](https://arxiv.org/abs/1905.00582)**

- **[Detecting GAN generated Fake Images using Co-occurrence Matrices](https://arxiv.org/abs/1903.06836)**

- **[Exposing DeepFake Videos By Detecting Face Warping Artifacts](https://arxiv.org/abs/1811.00656)**

- **[Protecting World Leaders Against Deep Fakes](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Agarwal_Protecting_World_Leaders_Against_Deep_Fakes_CVPRW_2019_paper.pdf)**

Software:

- **[DeepTrace](https://deeptracelabs.com/)**
