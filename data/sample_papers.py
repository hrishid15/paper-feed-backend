"""
Sample research paper data for testing the recommendation system
Papers span multiple ArXiv categories with realistic titles and abstracts
"""

SAMPLE_PAPERS = [
    # Machine Learning / AI Papers
    {
        "paper_id": "arxiv_2301_001",
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "authors": "Vaswani, Ashish; Shazeer, Noam; Parmar, Niki",
        "arxiv_category": "cs.LG",
        "keywords": "transformer, attention mechanism, neural networks, sequence modeling, NLP",
        "publication_date": "2017-06-12",
        "citation_count": 85000,
        "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf"
    },
    {
        "paper_id": "arxiv_2301_002",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
        "authors": "Devlin, Jacob; Chang, Ming-Wei; Lee, Kenton",
        "arxiv_category": "cs.CL",
        "keywords": "BERT, transformers, pre-training, language models, NLP, bidirectional encoding",
        "publication_date": "2018-10-11",
        "citation_count": 72000,
        "pdf_url": "https://arxiv.org/pdf/1810.04805.pdf"
    },
    {
        "paper_id": "arxiv_2301_003",
        "title": "Deep Residual Learning for Image Recognition",
        "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
        "authors": "He, Kaiming; Zhang, Xiangyu; Ren, Shaoqing; Sun, Jian",
        "arxiv_category": "cs.CV",
        "keywords": "ResNet, residual learning, deep learning, computer vision, image recognition, CNN",
        "publication_date": "2015-12-10",
        "citation_count": 95000,
        "pdf_url": "https://arxiv.org/pdf/1512.03385.pdf"
    },
    {
        "paper_id": "arxiv_2301_004",
        "title": "Generative Adversarial Networks",
        "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
        "authors": "Goodfellow, Ian; Pouget-Abadie, Jean; Mirza, Mehdi",
        "arxiv_category": "cs.LG",
        "keywords": "GAN, generative models, adversarial training, deep learning, neural networks",
        "publication_date": "2014-06-10",
        "citation_count": 58000,
        "pdf_url": "https://arxiv.org/pdf/1406.2661.pdf"
    },
    {
        "paper_id": "arxiv_2301_005",
        "title": "ImageNet Classification with Deep Convolutional Neural Networks",
        "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art.",
        "authors": "Krizhevsky, Alex; Sutskever, Ilya; Hinton, Geoffrey E.",
        "arxiv_category": "cs.CV",
        "keywords": "AlexNet, CNN, convolutional neural networks, ImageNet, computer vision, deep learning",
        "publication_date": "2012-09-30",
        "citation_count": 78000,
        "pdf_url": "https://arxiv.org/pdf/1409.0575.pdf"
    },
    
    # Reinforcement Learning
    {
        "paper_id": "arxiv_2301_006",
        "title": "Playing Atari with Deep Reinforcement Learning",
        "abstract": "We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.",
        "authors": "Mnih, Volodymyr; Kavukcuoglu, Koray; Silver, David",
        "arxiv_category": "cs.LG",
        "keywords": "deep Q-learning, reinforcement learning, DQN, game playing, neural networks",
        "publication_date": "2013-12-19",
        "citation_count": 15000,
        "pdf_url": "https://arxiv.org/pdf/1312.5602.pdf"
    },
    {
        "paper_id": "arxiv_2301_007",
        "title": "Mastering the Game of Go with Deep Neural Networks and Tree Search",
        "abstract": "The game of Go has long been viewed as the most challenging of classic games for artificial intelligence owing to its enormous search space and the difficulty of evaluating board positions and moves. Here we introduce a new approach to computer Go that uses value networks to evaluate board positions and policy networks to select moves.",
        "authors": "Silver, David; Huang, Aja; Maddison, Chris J.",
        "arxiv_category": "cs.AI",
        "keywords": "AlphaGo, Monte Carlo tree search, deep learning, reinforcement learning, game AI",
        "publication_date": "2016-01-27",
        "citation_count": 12000,
        "pdf_url": "https://arxiv.org/pdf/1703.01742.pdf"
    },
    
    # Computer Vision
    {
        "paper_id": "arxiv_2301_008",
        "title": "You Only Look Once: Unified, Real-Time Object Detection",
        "abstract": "We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation.",
        "authors": "Redmon, Joseph; Divvala, Santosh; Girshick, Ross; Farhadi, Ali",
        "arxiv_category": "cs.CV",
        "keywords": "YOLO, object detection, real-time detection, computer vision, CNN",
        "publication_date": "2015-06-08",
        "citation_count": 28000,
        "pdf_url": "https://arxiv.org/pdf/1506.02640.pdf"
    },
    {
        "paper_id": "arxiv_2301_009",
        "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "abstract": "There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.",
        "authors": "Ronneberger, Olaf; Fischer, Philipp; Brox, Thomas",
        "arxiv_category": "cs.CV",
        "keywords": "U-Net, image segmentation, biomedical imaging, convolutional networks, medical AI",
        "publication_date": "2015-05-18",
        "citation_count": 45000,
        "pdf_url": "https://arxiv.org/pdf/1505.04597.pdf"
    },
    
    # Natural Language Processing
    {
        "paper_id": "arxiv_2301_010",
        "title": "Sequence to Sequence Learning with Neural Networks",
        "abstract": "Deep Neural Networks (DNNs) are powerful models that have achieved excellent performance on difficult learning tasks. Although DNNs work well whenever large labeled training sets are available, they cannot be used to map sequences to sequences. In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure.",
        "authors": "Sutskever, Ilya; Vinyals, Oriol; Le, Quoc V.",
        "arxiv_category": "cs.CL",
        "keywords": "seq2seq, sequence learning, neural networks, LSTM, machine translation",
        "publication_date": "2014-09-10",
        "citation_count": 18000,
        "pdf_url": "https://arxiv.org/pdf/1409.3215.pdf"
    },
    {
        "paper_id": "arxiv_2301_011",
        "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
        "abstract": "Neural machine translation is a recently proposed approach to machine translation. Unlike the traditional statistical machine translation, the neural machine translation aims at building a single neural network that can be jointly tuned to maximize the translation performance. We conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder-decoder architecture.",
        "authors": "Bahdanau, Dzmitry; Cho, Kyunghyun; Bengio, Yoshua",
        "arxiv_category": "cs.CL",
        "keywords": "attention mechanism, neural machine translation, seq2seq, encoder-decoder, NLP",
        "publication_date": "2014-09-01",
        "citation_count": 35000,
        "pdf_url": "https://arxiv.org/pdf/1409.0473.pdf"
    },
    
    # Robotics
    {
        "paper_id": "arxiv_2301_012",
        "title": "End-to-End Training of Deep Visuomotor Policies",
        "abstract": "Policy search methods can allow robots to learn control policies for a wide range of tasks, but practical applications of policy search often require hand-engineered components for perception, state estimation, and low-level control. In this paper, we aim to answer the following question: does training the perception and control systems jointly end-to-end provide better performance than training each component separately?",
        "authors": "Levine, Sergey; Finn, Chelsea; Darrell, Trevor; Abbeel, Pieter",
        "arxiv_category": "cs.RO",
        "keywords": "visuomotor control, robot learning, end-to-end learning, deep learning, robotics",
        "publication_date": "2015-04-20",
        "citation_count": 2500,
        "pdf_url": "https://arxiv.org/pdf/1504.00702.pdf"
    },
    {
        "paper_id": "arxiv_2301_013",
        "title": "Learning Dexterous Manipulation for a Soft Robotic Hand from Human Demonstrations",
        "abstract": "We present an approach for learning dexterous manipulation skills for a soft robotic hand using human demonstrations. Our method combines imitation learning with reinforcement learning to enable a soft hand to perform complex manipulation tasks that would be difficult to program manually.",
        "authors": "Kumar, Vikash; Gupta, Abhishek; Todorov, Emanuel",
        "arxiv_category": "cs.RO",
        "keywords": "soft robotics, dexterous manipulation, imitation learning, robot learning, human demonstrations",
        "publication_date": "2016-09-15",
        "citation_count": 850,
        "pdf_url": "https://arxiv.org/pdf/1609.01387.pdf"
    },
    
    # Quantum Computing
    {
        "paper_id": "arxiv_2301_014",
        "title": "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices",
        "abstract": "The Quantum Approximate Optimization Algorithm (QAOA) is a promising approach for solving combinatorial optimization problems on near-term quantum computers. We present a comprehensive analysis of QAOA performance and provide insights into the mechanisms that enable QAOA to find approximate solutions to hard optimization problems.",
        "authors": "Zhou, Leo; Wang, Sheng-Tao; Choi, Soonwon",
        "arxiv_category": "quant-ph",
        "keywords": "QAOA, quantum computing, optimization, quantum algorithms, NISQ",
        "publication_date": "2018-12-10",
        "citation_count": 450,
        "pdf_url": "https://arxiv.org/pdf/1812.01041.pdf"
    },
    {
        "paper_id": "arxiv_2301_015",
        "title": "Variational Quantum Eigensolver: A Review",
        "abstract": "The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm designed to find the ground state energy of a quantum system. This review covers the theoretical foundations, implementation details, and applications of VQE on near-term quantum devices.",
        "authors": "Cerezo, Marco; Arrasmith, Andrew; Babbush, Ryan",
        "arxiv_category": "quant-ph",
        "keywords": "VQE, variational quantum eigensolver, quantum chemistry, hybrid algorithms, quantum computing",
        "publication_date": "2020-03-15",
        "citation_count": 380,
        "pdf_url": "https://arxiv.org/pdf/2003.08063.pdf"
    },
    
    # Theoretical CS / Algorithms
    {
        "paper_id": "arxiv_2301_016",
        "title": "Graph Neural Networks: A Review of Methods and Applications",
        "abstract": "Lots of learning tasks require dealing with graph data which contains rich relation information among elements. Graph neural networks (GNNs) have emerged as a powerful approach for representation learning on graphs. This paper provides a comprehensive overview of GNNs, including the mechanisms, variants, and applications.",
        "authors": "Zhou, Jie; Cui, Ganqu; Hu, Shengding; Zhang, Zhengyan",
        "arxiv_category": "cs.LG",
        "keywords": "graph neural networks, GNN, graph representation learning, node classification, link prediction",
        "publication_date": "2018-12-20",
        "citation_count": 5200,
        "pdf_url": "https://arxiv.org/pdf/1812.08434.pdf"
    },
    {
        "paper_id": "arxiv_2301_017",
        "title": "Adam: A Method for Stochastic Optimization",
        "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters.",
        "authors": "Kingma, Diederik P.; Ba, Jimmy",
        "arxiv_category": "cs.LG",
        "keywords": "Adam optimizer, stochastic optimization, gradient descent, deep learning, optimization algorithms",
        "publication_date": "2014-12-22",
        "citation_count": 95000,
        "pdf_url": "https://arxiv.org/pdf/1412.6980.pdf"
    },
    
    # Multimodal / Vision-Language
    {
        "paper_id": "arxiv_2301_018",
        "title": "Learning Transferable Visual Models From Natural Language Supervision",
        "abstract": "State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision.",
        "authors": "Radford, Alec; Kim, Jong Wook; Hallacy, Chris",
        "arxiv_category": "cs.CV",
        "keywords": "CLIP, vision-language models, multimodal learning, zero-shot learning, contrastive learning",
        "publication_date": "2021-02-26",
        "citation_count": 8500,
        "pdf_url": "https://arxiv.org/pdf/2103.00020.pdf"
    },
    {
        "paper_id": "arxiv_2301_019",
        "title": "Flamingo: a Visual Language Model for Few-Shot Learning",
        "abstract": "Building models that can be rapidly adapted to novel tasks using only a handful of annotated examples is an open challenge for multimodal machine learning research. We introduce Flamingo, a family of Visual Language Models (VLM) with this ability. Flamingo models can be applied to tasks using only a few input/output examples.",
        "authors": "Alayrac, Jean-Baptiste; Donahue, Jeff; Luc, Pauline",
        "arxiv_category": "cs.CV",
        "keywords": "visual language models, few-shot learning, multimodal AI, vision and language, in-context learning",
        "publication_date": "2022-04-29",
        "citation_count": 1200,
        "pdf_url": "https://arxiv.org/pdf/2204.14198.pdf"
    },
    
    # Diffusion Models
    {
        "paper_id": "arxiv_2301_020",
        "title": "Denoising Diffusion Probabilistic Models",
        "abstract": "We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics.",
        "authors": "Ho, Jonathan; Jain, Ajay; Abbeel, Pieter",
        "arxiv_category": "cs.LG",
        "keywords": "diffusion models, generative models, image synthesis, DDPM, denoising",
        "publication_date": "2020-06-19",
        "citation_count": 6800,
        "pdf_url": "https://arxiv.org/pdf/2006.11239.pdf"
    },

    # More NLP/Language Models
    {
        "paper_id": "arxiv_2301_021",
        "title": "GPT-3: Language Models are Few-Shot Learners",
        "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance.",
        "authors": "Brown, Tom B.; Mann, Benjamin; Ryder, Nick",
        "arxiv_category": "cs.CL",
        "keywords": "GPT-3, language models, few-shot learning, in-context learning, large language models",
        "publication_date": "2020-05-28",
        "citation_count": 45000,
        "pdf_url": "https://arxiv.org/pdf/2005.14165.pdf"
    },
    {
        "paper_id": "arxiv_2301_022",
        "title": "XLNet: Generalized Autoregressive Pretraining for Language Understanding",
        "abstract": "With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining approaches based on autoregressive language modeling. However, relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. We propose XLNet, a generalized autoregressive pretraining method.",
        "authors": "Yang, Zhilin; Dai, Zihang; Yang, Yiming",
        "arxiv_category": "cs.CL",
        "keywords": "XLNet, autoregressive models, language understanding, pre-training, permutation language modeling",
        "publication_date": "2019-06-19",
        "citation_count": 8500,
        "pdf_url": "https://arxiv.org/pdf/1906.08237.pdf"
    },
    
    # More Computer Vision
    {
        "paper_id": "arxiv_2301_023",
        "title": "Mask R-CNN",
        "abstract": "We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition.",
        "authors": "He, Kaiming; Gkioxari, Georgia; Dollár, Piotr; Girshick, Ross",
        "arxiv_category": "cs.CV",
        "keywords": "instance segmentation, Mask R-CNN, object detection, computer vision, deep learning",
        "publication_date": "2017-03-20",
        "citation_count": 32000,
        "pdf_url": "https://arxiv.org/pdf/1703.06870.pdf"
    },
    {
        "paper_id": "arxiv_2301_024",
        "title": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
        "abstract": "Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance.",
        "authors": "Tan, Mingxing; Le, Quoc V.",
        "arxiv_category": "cs.CV",
        "keywords": "EfficientNet, model scaling, neural architecture, computer vision, image classification",
        "publication_date": "2019-05-28",
        "citation_count": 18000,
        "pdf_url": "https://arxiv.org/pdf/1905.11946.pdf"
    },
    {
        "paper_id": "arxiv_2301_025",
        "title": "Vision Transformer: An Image is Worth 16x16 Words",
        "abstract": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.",
        "authors": "Dosovitskiy, Alexey; Beyer, Lucas; Kolesnikov, Alexander",
        "arxiv_category": "cs.CV",
        "keywords": "vision transformer, ViT, self-attention, image classification, transformers for vision",
        "publication_date": "2020-10-22",
        "citation_count": 25000,
        "pdf_url": "https://arxiv.org/pdf/2010.11929.pdf"
    },
    
    # More Reinforcement Learning
    {
        "paper_id": "arxiv_2301_026",
        "title": "Proximal Policy Optimization Algorithms",
        "abstract": "We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a surrogate objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates.",
        "authors": "Schulman, John; Wolski, Filip; Dhariwal, Prafulla",
        "arxiv_category": "cs.LG",
        "keywords": "PPO, proximal policy optimization, reinforcement learning, policy gradient, deep RL",
        "publication_date": "2017-07-20",
        "citation_count": 12000,
        "pdf_url": "https://arxiv.org/pdf/1707.06347.pdf"
    },
    {
        "paper_id": "arxiv_2301_027",
        "title": "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning",
        "abstract": "Model-free deep reinforcement learning (RL) algorithms have been demonstrated on a range of challenging decision making and control tasks. However, these methods typically suffer from two major challenges: very high sample complexity and brittle convergence properties. We propose soft actor-critic, an off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework.",
        "authors": "Haarnoja, Tuomas; Zhou, Aurick; Abbeel, Pieter; Levine, Sergey",
        "arxiv_category": "cs.LG",
        "keywords": "SAC, soft actor-critic, maximum entropy RL, off-policy learning, deep reinforcement learning",
        "publication_date": "2018-01-12",
        "citation_count": 8500,
        "pdf_url": "https://arxiv.org/pdf/1801.01290.pdf"
    },
    
    # More Generative Models
    {
        "paper_id": "arxiv_2301_028",
        "title": "Variational Autoencoders",
        "abstract": "How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets? We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case.",
        "authors": "Kingma, Diederik P.; Welling, Max",
        "arxiv_category": "cs.LG",
        "keywords": "VAE, variational autoencoder, generative models, latent variables, variational inference",
        "publication_date": "2013-12-20",
        "citation_count": 28000,
        "pdf_url": "https://arxiv.org/pdf/1312.6114.pdf"
    },
    {
        "paper_id": "arxiv_2301_029",
        "title": "StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks",
        "abstract": "We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes and stochastic variation in the generated images, and it enables intuitive, scale-specific control of the synthesis.",
        "authors": "Karras, Tero; Laine, Samuli; Aila, Timo",
        "arxiv_category": "cs.CV",
        "keywords": "StyleGAN, generative adversarial networks, style transfer, image synthesis, GAN",
        "publication_date": "2018-12-12",
        "citation_count": 15000,
        "pdf_url": "https://arxiv.org/pdf/1812.04948.pdf"
    },
    {
        "paper_id": "arxiv_2301_030",
        "title": "Stable Diffusion: High-Resolution Image Synthesis with Latent Diffusion Models",
        "abstract": "By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis quality on image data. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and requires massive datasets. To enable training on limited computational resources, we apply diffusion models in the latent space of powerful pretrained autoencoders.",
        "authors": "Rombach, Robin; Blattmann, Andreas; Lorenz, Dominik",
        "arxiv_category": "cs.CV",
        "keywords": "stable diffusion, latent diffusion, image synthesis, text-to-image, diffusion models",
        "publication_date": "2021-12-20",
        "citation_count": 9500,
        "pdf_url": "https://arxiv.org/pdf/2112.10752.pdf"
    },
    
    # Graph Learning
    {
        "paper_id": "arxiv_2301_031",
        "title": "Graph Attention Networks",
        "abstract": "We present graph attention networks (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods' features, we enable (implicitly) specifying different weights to different nodes in a neighborhood.",
        "authors": "Veličković, Petar; Cucurull, Guillem; Casanova, Arantxa",
        "arxiv_category": "cs.LG",
        "keywords": "graph attention networks, GAT, graph neural networks, attention mechanism, node classification",
        "publication_date": "2017-10-30",
        "citation_count": 12000,
        "pdf_url": "https://arxiv.org/pdf/1710.10903.pdf"
    },
    
    # Meta-Learning
    {
        "paper_id": "arxiv_2301_032",
        "title": "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks",
        "abstract": "We propose an algorithm for meta-learning that is model-agnostic, in the sense that it is compatible with any model trained with gradient descent and applicable to a variety of different learning problems, including classification, regression, and reinforcement learning. The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples.",
        "authors": "Finn, Chelsea; Abbeel, Pieter; Levine, Sergey",
        "arxiv_category": "cs.LG",
        "keywords": "MAML, meta-learning, few-shot learning, transfer learning, neural networks",
        "publication_date": "2017-03-09",
        "citation_count": 9800,
        "pdf_url": "https://arxiv.org/pdf/1703.03400.pdf"
    },
    
    # Explainable AI
    {
        "paper_id": "arxiv_2301_033",
        "title": "LIME: Local Interpretable Model-agnostic Explanations",
        "abstract": "Despite widespread adoption, machine learning models remain mostly black boxes. Understanding the reasons behind predictions is, however, quite important in assessing trust, which is fundamental if one plans to take action based on a prediction, or when choosing whether to deploy a new model. We propose LIME, a novel explanation technique that explains the predictions of any classifier in an interpretable and faithful manner.",
        "authors": "Ribeiro, Marco Tulio; Singh, Sameer; Guestrin, Carlos",
        "arxiv_category": "cs.LG",
        "keywords": "LIME, explainable AI, interpretability, model explanations, XAI",
        "publication_date": "2016-02-16",
        "citation_count": 11000,
        "pdf_url": "https://arxiv.org/pdf/1602.04938.pdf"
    },
    
    # Federated Learning
    {
        "paper_id": "arxiv_2301_034",
        "title": "Communication-Efficient Learning of Deep Networks from Decentralized Data",
        "abstract": "Modern mobile devices have access to a wealth of data suitable for learning models, which in turn can greatly improve the user experience on the device. For example, language models can improve speech recognition and text entry, and image models can automatically select good photos. However, this rich data is often privacy sensitive, large in quantity, or both, which may preclude logging to the data center and training there using conventional approaches.",
        "authors": "McMahan, Brendan; Moore, Eider; Ramage, Daniel",
        "arxiv_category": "cs.LG",
        "keywords": "federated learning, privacy-preserving ML, distributed learning, on-device learning, communication efficiency",
        "publication_date": "2016-02-17",
        "citation_count": 8200,
        "pdf_url": "https://arxiv.org/pdf/1602.05629.pdf"
    },
    
    # Self-Supervised Learning
    {
        "paper_id": "arxiv_2301_035",
        "title": "Momentum Contrast for Unsupervised Visual Representation Learning",
        "abstract": "We present Momentum Contrast (MoCo) for unsupervised visual representation learning. From a perspective on contrastive learning as dictionary look-up, we build a dynamic dictionary with a queue and a moving-averaged encoder. This enables building a large and consistent dictionary on-the-fly that facilitates contrastive unsupervised learning.",
        "authors": "He, Kaiming; Fan, Haoqi; Wu, Yuxin; Xie, Saining; Girshick, Ross",
        "arxiv_category": "cs.CV",
        "keywords": "MoCo, momentum contrast, self-supervised learning, contrastive learning, unsupervised learning",
        "publication_date": "2019-11-13",
        "citation_count": 10500,
        "pdf_url": "https://arxiv.org/pdf/1911.05722.pdf"
    }
]

# Sample user interactions for testing
SAMPLE_INTERACTIONS = [
    # User interested in NLP/Transformers
    {"user_id": "researcher_alice", "paper_id": "arxiv_2301_001", "interaction_type": "save", "rating": 5, "time_spent_seconds": 420},
    {"user_id": "researcher_alice", "paper_id": "arxiv_2301_002", "interaction_type": "like", "rating": 5, "time_spent_seconds": 380},
    {"user_id": "researcher_alice", "paper_id": "arxiv_2301_011", "interaction_type": "save", "rating": 4, "time_spent_seconds": 300},
    
    # User interested in Computer Vision
    {"user_id": "researcher_bob", "paper_id": "arxiv_2301_003", "interaction_type": "save", "rating": 5, "time_spent_seconds": 450},
    {"user_id": "researcher_bob", "paper_id": "arxiv_2301_005", "interaction_type": "like", "rating": 5, "time_spent_seconds": 360},
    {"user_id": "researcher_bob", "paper_id": "arxiv_2301_008", "interaction_type": "save", "rating": 4, "time_spent_seconds": 280},
    {"user_id": "researcher_bob", "paper_id": "arxiv_2301_009", "interaction_type": "like", "rating": 4, "time_spent_seconds": 200},
    
    # User interested in Robotics/RL
    {"user_id": "researcher_carol", "paper_id": "arxiv_2301_006", "interaction_type": "save", "rating": 5, "time_spent_seconds": 400},
    {"user_id": "researcher_carol", "paper_id": "arxiv_2301_007", "interaction_type": "like", "rating": 5, "time_spent_seconds": 500},
    {"user_id": "researcher_carol", "paper_id": "arxiv_2301_012", "interaction_type": "save", "rating": 4, "time_spent_seconds": 320},
    {"user_id": "researcher_carol", "paper_id": "arxiv_2301_013", "interaction_type": "like", "rating": 4, "time_spent_seconds": 250},

    # User interested in Generative Models
    {"user_id": "researcher_eve", "paper_id": "arxiv_2301_004", "interaction_type": "save", "rating": 5, "time_spent_seconds": 500},
    {"user_id": "researcher_eve", "paper_id": "arxiv_2301_020", "interaction_type": "like", "rating": 5, "time_spent_seconds": 450},
    {"user_id": "researcher_eve", "paper_id": "arxiv_2301_028", "interaction_type": "save", "rating": 4, "time_spent_seconds": 380},
    {"user_id": "researcher_eve", "paper_id": "arxiv_2301_029", "interaction_type": "like", "rating": 4, "time_spent_seconds": 320},
    
    # User interested in Meta-Learning
    {"user_id": "researcher_frank", "paper_id": "arxiv_2301_032", "interaction_type": "save", "rating": 5, "time_spent_seconds": 420},
    {"user_id": "researcher_frank", "paper_id": "arxiv_2301_002", "interaction_type": "like", "rating": 4, "time_spent_seconds": 300},
]

# Add to USER_INTERESTS
USER_INTERESTS = {
    "researcher_alice": ["natural language processing", "transformers", "attention mechanisms", "language models"],
    "researcher_bob": ["computer vision", "convolutional neural networks", "object detection", "image segmentation"],
    "researcher_carol": ["reinforcement learning", "robotics", "robot learning", "visuomotor control"],
    "researcher_dave": ["quantum computing", "quantum algorithms", "optimization"],
    "researcher_eve": ["generative models", "diffusion models", "image synthesis", "GANs"],
    "researcher_frank": ["meta-learning", "few-shot learning", "transfer learning"]
}