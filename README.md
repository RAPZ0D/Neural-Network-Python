# Building a Neural Networks Using Python Programming Language

- [What is a Neural Network](#what-is-a-neural-network)
- 
- [Libraries for building a Neural Network](#libraries-for-building-a-neural-network)
- 
- [Artificial Neural Network](#artificial-neural-network)
- [Convolution Neural Network](#convolution-neural-network)
- [Recurrent Neural Network](#recurrent-neural-network)


## What is a Neural Network
What is Neural Networks?
Neural Networks are computational systems inspired by the human brain's structure and function. They consist of interconnected nodes, or neurons, organized into layers that process information.

Neurons and Layers:
1. Neurons: These are the basic units that receive inputs, apply weights to them, and pass the result through an activation function.
2. Layers: Neurons are organized into layers—input, hidden, and output layers. Input layer receives initial data, hidden layers process information, and output layer produces the final result.

Some of its applications are: 
1. Image and Speech Recognition: CNNs excel in recognizing patterns and features within images and audio.
2. Natural Language Processing (NLP): RNNs are effective in processing sequences of words for tasks like language translation, sentiment analysis, and text generation.
3. Predictive Analysis and Forecasting: Neural networks are used extensively in financial forecasting, trend analysis, and predictive modeling.

**Types of Neural Netowrks:**
1. Artificial Neural Netowrks (ANN)
2. Convolution Neural Netowkr (CNN)
3. Recurrent Neural Networks (RNN)


Here is a very informative video about Neural Networks. It is a video by 3Blue1Brown [What is Neural Network](https://www.youtube.com/watch?v=aircAruvnKk)
The Image Below is a representation of what a Neural Network is in simple terms
So Neural Network is inspired by the biological neuron in a simplified manner. A biological neuron is a cell that processes and transmits information in the brain. It consists of a cell body, dendrites, and an axon. Dendrites receive signals, the cell body processes these signals, and the axon transmits signals to other neurons.

![neural](https://github.com/RAPZ0D/Neural-Network-Python/assets/100001521/24f148d8-f2e9-4fac-b236-1eeb4e5c6dd6)


**The MATH behind Neural Networks**

Neural Networks in Machine Learning is a combination of Biology and Mathematics. Math is the key language behind neural networks, helping them understand and process information efficiently. It's crucial because it enables these networks to describe how neurons work together, perform calculations quickly, and learn from examples. Math also guides the fine-tuning of these networks, ensuring they make accurate predictions and generalize well to new, unseen data. It's like the foundation that allows neural networks to do their job—making sense of data, learning from it, and making useful decisions or predictions. Without math, it would be really tough for these networks to work effectively and learn from the world around them.
Here's an overview along with some key mathematical elements:

    1. Linear Transformation:
        Formula: z=w⋅x+bz=w⋅x+b
        This represents the core operation within a neuron where xx is the input, ww is the weight, bb is the bias, and zz is the weighted sum.
        Matrix Form: For multiple inputs and weights, this operation can be represented as matrix multiplication: Z=W⋅X+BZ=W⋅X+B, where ZZ and XX are matrices of weighted sums.

    2. Activation Functions:
        Formula (Sigmoid): σ(z)=11+e−zσ(z)=1+e−z1​
        Activation functions introduce non-linearity, enabling neural networks to model complex relationships in data.
        Other Functions: ReLU (Rectified Linear Unit), Tanh, Leaky ReLU, etc., each with its own mathematical formulation.

    3. Loss Functions:
        Formula (Mean Squared Error): L(y,y^)=1n∑i=1n(yi−y^i)2L(y,y^​)=n1​∑i=1n​(yi​−y^​i​)2
        These functions measure the difference between predicted (y^y^​) and actual (yy) values, guiding the learning process by quantifying the model's performance.

    4. Gradient Descent:
        Formula (Gradient Descent Update Rule): wi+1=wi−α⋅∂L∂wwi+1​=wi​−α⋅∂w∂L​
        Used in training, this formula updates weights (ww) based on the gradient of the loss function (LL) with respect to the weights, multiplied by a learning rate (αα).

    5. Backpropagation:
        Involves the application of the chain rule of calculus to compute gradients throughout the network. It calculates how changing each weight in the network contributes to the overall error.

    6. Matrix Calculus:
       Crucial for efficiently computing gradients in neural networks. Involves operations like matrix differentiations (e.g., calculating gradients with respect to weight matrices in various layers).

 ![math_neural](https://github.com/RAPZ0D/Neural-Network-Python/assets/100001521/9442ab8c-066a-495c-89b6-ca2dee9d4830)

 
 **The Activation Functions**
 
![graphs](https://github.com/RAPZ0D/Neural-Network-Python/assets/100001521/25cd6677-b553-46b7-b269-42d86c606176)


## Libraries for building a Neural Network

When developing neural networks in Python, several key libraries play integral roles in the construction, training, and evaluation of models. Here are the fundamental libraries used for building neural networks:

1. **TensorFlow / Keras**:
   - **TensorFlow**: A powerful open-source machine learning library developed by Google Brain. It provides a flexible ecosystem for building and deploying machine learning models, especially neural networks.
   - **Keras**: A high-level neural networks API, often used as an interface on top of TensorFlow. It enables rapid prototyping and experimentation, offering ease of use and flexibility.

2. **PyTorch**:
   - Developed by Facebook's AI Research lab, PyTorch is another widely used open-source deep learning library. It's known for its dynamic computation graph and simplicity in building neural networks.

3. **NumPy**:
   - NumPy is a fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, essential for handling data in neural networks.

4. **Pandas**:
   - Pandas is a powerful library used for data manipulation and analysis. It provides data structures like DataFrames, which are helpful for preprocessing and organizing data for neural network training.

5. **Matplotlib / Seaborn**:
   - **Matplotlib**: A plotting library for Python that helps in visualizing data, plotting graphs, and displaying neural network architectures or training progress.
   - **Seaborn**: Another data visualization library built on top of Matplotlib, offering enhanced aesthetics and statistical plotting.

6. **Scikit-learn**:
   - Though primarily known for machine learning algorithms, Scikit-learn provides utilities for preprocessing data, splitting datasets, and evaluation metrics that can complement neural network workflows.

### Role and Importance
These libraries collectively provide a comprehensive toolkit for building, training, and deploying neural networks. They facilitate various tasks such as model creation, data handling, visualization, and evaluation, streamlining the development process and enhancing the capabilities of neural network-based applications.


## Artificial Neural Network 
Artificial Neural Networks (ANNs) are computational models inspired by the human brain's neural structure. They consist of interconnected nodes (neurons) arranged in layers, including input, hidden, and output layers. ANNs process information in a way that allows them to learn patterns and relationships within data. ANNs mimic the brain's interconnected neurons, where each neuron receives inputs, processes them, and produces an output. Similarly, in ANNs, nodes receive inputs, apply weights to these inputs, sum them up with biases, and pass the result through an activation function to generate an output. The network learns by adjusting these weights and biases through training, optimizing its ability to make accurate predictions or classifications.

Its Uses are: 
1. Pattern Recognition: ANNs excel in recognizing patterns within data, making them valuable for image and speech recognition tasks. They can detect features, classify objects, and recognize speech patterns.
2. Predictive Analysis and Forecasting: These networks are used extensively in predictive modeling. In finance, they're employed for stock market prediction, in weather forecasting for climate modeling, and in various industries for demand forecasting.
3. Natural Language Processing (NLP): ANNs play a crucial role in NLP tasks like language translation, sentiment analysis, and text generation. They help in understanding and processing human language.

The benifits of using ANNs are: 
- Adaptability and Learning: ANNs can learn from large datasets, adapt to new information, and improve their performance over time. They can handle complex, nonlinear relationships in data.

- Parallel Processing: They can process information simultaneously across nodes, allowing for parallel computation and faster processing of tasks compared to traditional algorithms.

- Generalization: ANNs can generalize patterns from the data they've been trained on, making them applicable to new, unseen data—a crucial aspect for real-world applications.

**A Visualization of ANN**

![nn1](https://github.com/RAPZ0D/Neural-Network-Python/assets/100001521/55482176-4dd9-4bbd-9f44-b2a7654fa403)


__Types of ANN__
![types](https://github.com/RAPZ0D/Neural-Network-Python/assets/100001521/729bb417-be2c-4398-b5d5-16fcc95a17cb)


## Convolution Neural Network

- **Explanation of Convolutional Neural Networks (CNNs)**:
  - CNNs leverage specialized layers, including:
    - *Convolutional Layers*: Apply filters to extract features like edges and textures.
    - *Pooling Layers*: Downsample features while retaining crucial information.
    - *Fully Connected Layers*: Perform classification based on learned features.

- **Uses and Applications**:
  1. *Image Classification*: Accurately categorize images into various classes.
  2. *Object Detection*: Identify and localize multiple objects within an image.
  3. *Image Segmentation*: Segment images into meaningful parts, crucial in medical imaging.
  4. *Feature Extraction*: Extract learned features for transfer learning.

- **Math Behind CNNs**:
  - **Convolution Operation**: Element-wise multiplication and summation between filters and image patches.
  - **Pooling Operations**: Max pooling or average pooling for downsampling.
  - **Activation Functions**: Introduce non-linearity in CNN layers.

CNNs excel in processing structured data like images, learning hierarchical representations, and extracting intricate features—fundamental in modern computer vision applications.

The Images will Explain how the Convolution Neural Network works:

**Architecture of CNNs**

![convo](https://github.com/RAPZ0D/Neural-Network-Python/assets/100001521/ca329bca-39bc-452d-8fec-70908566d03d)


![convo2](https://github.com/RAPZ0D/Neural-Network-Python/assets/100001521/6659679f-952d-4804-a5d2-e125a49d0cdd)


**Classification using CNN**

![convo3](https://github.com/RAPZ0D/Neural-Network-Python/assets/100001521/8fcff785-85c9-4eac-9bdc-9901cad432bc)


## Recurrent Neural Network

### Recurrent Neural Networks (RNNs)

- **Explanation of Recurrent Neural Networks (RNNs)**:
  - RNNs process sequential data by maintaining a hidden state that evolves as new inputs are fed in. They utilize recurrent connections to retain context from previous inputs.
  - Components include a hidden state, recurrent connections, and gate mechanisms (e.g., LSTM, GRU) that enhance long-term dependencies.

- **Uses and Applications**:
  1. *Natural Language Processing (NLP)*: Crucial in language modeling, translation, sentiment analysis, and text generation due to their sequential data processing capability.
  2. *Time Series Prediction*: Used in financial forecasting, weather prediction, and any domain involving sequential data.
  3. *Speech Recognition*: Applied in systems processing sequences of audio signals for speech recognition.
  4. *Video Analysis*: Used in video classification, action recognition, and video description tasks.

- **Building Blocks**:
  - *Vanishing Gradient*: RNNs may suffer from the vanishing gradient problem, affecting their ability to learn long-term dependencies efficiently.
  - *Long Short-Term Memory (LSTM)*: Addresses vanishing gradient by introducing gating mechanisms for better learning of long-term dependencies.
  - *Gated Recurrent Unit (GRU)*: A simplified variation of LSTM, maintaining advantages in handling long-term dependencies.

- **Mathematical Aspects**:
  - **Hidden State Evolution**:
    - At each time step \( t \), the hidden state \( h_t \) is updated based on the input \( x_t \) and the previous hidden state \( h_{t-1} \) using a function \( f \) with parameters \( W \):
      \[ h_t = f(x_t, h_{t-1}; W) \]
  - **Backpropagation Through Time (BPTT)**:
    - Derivatives are computed through the unfolded network in time, allowing gradients to be propagated backward. However, vanishing gradients can affect long-term dependencies.
  - **Long Short-Term Memory (LSTM) Math**:
    - LSTM introduces gates (input, forget, output) controlled by sigmoid and tanh functions to regulate information flow, addressing the vanishing gradient issue. It updates the cell state \( C_t \) and hidden state \( h_t \) based on inputs and previous states.
  - **Gated Recurrent Unit (GRU) Math**:
   - GRU simplifies the LSTM architecture by merging the cell state and hidden state, using reset and update gates to control the flow of information.


RNNs are powerful tools for processing sequential data, retaining memory across time steps, and finding applications in diverse domains requiring sequential modeling and understanding.

Visualizing the RNN: An illustrative diagram providing a graphical insight into the structure and flow of Recurrent Neural Networks (RNNs).

<div align="center"> <img src="https://github.com/RAPZ0D/Neural-Network-Python/assets/100001521/2409a405-d601-45c2-aba5-8b5fd114b6ea"> </div>


<p align="center"> <img src="https://github.com/RAPZ0D/Neural-Network-Python/assets/100001521/8eda07e2-779b-4990-9648-4e4773cb2076"> </p>

