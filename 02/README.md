# Machine Learning / Neural Networks

## Supervised Machine Learning: Regression and Classification
Supervised learning is like teaching a child with flashcards - you provide examples with the correct answers, and the model learns to recognize patterns and make predictions based on that training.
The key idea is that `we have labeled data` - each input comes with the right output, and the model's job is to learn the mapping from inputs to outputs.
### Types of Supervised Learning
- **Regression**: Predicting continuous outcomes (e.g., house prices, temperatures).
- **Classification**: Predicting discrete outcomes (e.g., spam detection, image recognition).

### Linear Regression
Covers: Simple and multiple linear regression, cost function, gradient descent, Mean Squared Error (MSE).
Linear regression for predicting continuous outcomes
[🧪Energy consumption prediction](energy-consumption.ipynb)
[🧪Medical insurance costs prediction](medical-costs.ipynb)

#### Power Law case study
Power law is a functional relationship where a relative change in one quantity results in a proportional relative change in another quantity, often expressed as $y = k * x^a$, where k and a are constants.
`Kleiber's law` is a specific application of power law in biology, stating that an organism's metabolic rate scales to the 3/4 power of its mass $B = k * M^{(3/4)}$, indicating that larger organisms are more energy-efficient than smaller ones.
[🧪Lab](./Kleiber-law.ipynb)
[📖Wiki](https://en.wikipedia.org/wiki/Power_law)
[📖Kleiber's law](https://en.wikipedia.org/wiki/Kleiber%27s_law)
[📖Scale: the universal laws of life and death in organisms, cities and companies](https://www.amazon.it/dp/1780225598/)


### Logistic Regression
Logistic regression for predicting categorical outcomes.
Covers: Binary classification, decision boundary, performance metrics (accuracy, precision, recall).
Labs: sentiment analysis
[🧪Amazon Reviews Sentiment Analysis](amazon-reviews-sentiment.ipynb)

#### Metrics to evaluate model performance
Covers: Accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC.
Labs: Evaluate classification model performance
**Accuracy**: how often a  model correctly predicts the outcome<br/>(Number of Correct Predictions) / (Total Number of Predictions) or $(TP + TN) / (TP + TN + FP + FN)$
Used when classes are balanced (has roughly the same number of instances)
**Precision**: When the model says something is true, how often is it actually true? =  $TP / (TP + FP)$
Used in unbalanced class scenarios where false positives are costly (e.g., spam detection, where you don't want to misclassify important emails as spam)
**Recall**: Of all the actual positive instances, how many did the model correctly predict as positive? = $TP / (TP + FN)$
Used in unbalanced class scenarios where false negatives are costly (e.g., disease detection, where you don't want to miss actual cases)
[🔗Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
**F1-Score**: Balancing precision and recall = $2 * (Precision * Recall) / (Precision + Recall)$
Used when you need a single metric that considers both precision and recall, where the costs of false positives and false negatives are equally important (e.g. information retrieval, search engines)
**Confusion Matrix**: Table showing TP, TN, FP, FN counts
Used to visualize model performance across different classes
**ROC-AUC**: Area under the ROC curve, measuring trade-off between TPR and FPR
Used to evaluate model's ability to distinguish between classes
**Log Loss**:  $-1/N * Σ [y * log(p) + (1 - y) * log(1 - p)]$
Measures the uncertainty of predictions based on probability estimates

## Unsupervised Machine Learning: Clustering and Dimensionality Reduction
Unsupervised learning is like exploring a new city without a map - the model looks for patterns and structures in the data without any labeled examples to guide it.
The key idea is that `we don’t have labeled data` - the model must find its own way to group similar items or reduce complexity.
An algorithm learns the structure of the data without any assistance. This creates two main differences. 
1. Allows us to process `large amounts of data` because the data does not need to be manually labeled. 
2. It is `difficult to evaluate` the quality of an unsupervised algorithm due to the absence of an explicit goodness metric as used in supervised learning.

### Types of Unsupervised Learning
1. **Clustering**: Grouping similar data points together (e.g., customer segmentation for recommendations).
2. **Dimensionality Reduction**: Reducing the number of features while preserving important information (e.g., PCA: principal component analysis).
3. **Anomaly Detection**: Identifying unusual data points that differ significantly from the majority (e.g., fraud detection).

#### K-Means Clustering
  K-means clustering for grouping similar data points
  K-means algorithm, choosing the number of clusters (k), inertia, silhouette score.
  [🧪Iris dataset](iris.ipynb)

## Reinforcement Learning
Reinforcement learning is like teaching through `trial and error` - an agent learns by trying actions, receiving feedback (**rewards**), and gradually improving its behavior. Think of training a pet with treats, learning to ride a bike through practice, or mastering a video game by playing it repeatedly.

The key insight is that `we don’t tell the agent exactly what to do`. Instead, we create an **environment** where it can experiment safely and learn from the consequences of its actions.

#### Agent-Environment Loop

<details>
The agent-environment loop is the core of reinforcement learning. It works like this:

- Agent observes the current situation (like looking at a game screen)

- Agent chooses an action based on what it sees (like pressing a button)

- Environment responds with a new situation and a reward (game state changes, score updates)

Repeat until the episode ends

![A-E Loop Image](./AE_loop.png)
<sub>Image ©Gymnasium Farama</sub>
[🧪Demo blackjack](./blackjack.ipynb)
[🔗Gymnasium RL](https://gymnasium.farama.org/)
[🧪Intro to Game AI and Reinforcement Learning](https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning)
</details>

--- 

## Neural Networks

A neural network is a computational model inspired by the structure and function of biological neural networks. 
It consists of interconnected nodes, called `neurons`, organized in `layers`. 
These neurons process and transmit signals, learning complex patterns from data through adjusting the strengths of the connections (`weights`) between them.
[📺NN in 5 minutes](https://www.youtube.com/watch?v=jmmW0F0biz0)

### NN types

- **Feedforward Neural Networks (FNNs)**: Information flows in one direction from input to output. No cycles or loops. Sample applications include image classification and regression tasks.
- **Convolutional Neural Networks (CNNs)**: Specialized for processing grid-like data, such as images. Sample applications include image recognition and computer vision.
- **Recurrent Neural Networks (RNNs)**: Designed for sequential data, with connections that form cycles. Sample applications include language modeling and time series prediction.

### basic concepts

- **neuron**: basic unit that receives input, processes it, and produces output.
- **layer**: group of neurons that process inputs and pass outputs to the next layer. 
- **input layer**: first layer that receives raw data.
- **hidden layers**: intermediate layers that extract features and patterns. Hidden layers enable the network to learn complex representations of the input data by transforming it through multiple stages. Common layer types include:
  - `Dense` layer (fully connected, used in FNNs), can be in input, hidden, or output layers
  - `Convolutional` layer (used in CNNs), to process data with spatial hierarchy, like images
  - `Recurrent` layer (used in RNNs), to process sequential data, like text or time series
  - `Flatten` layer (to convert multi-dimensional input into a 1D vector, commonly used before fully connected layers), i.e. transforming a 2D image into a 1D array for processing in dense layers.
  - `Batch Normalization` layer (to normalize inputs of each layer, improving training speed and stability), i.e scaling image pixel values to a standard range before feeding them into the network.
  - `Dropout` layer (with randomly deactivated neurons during training to prevent overfitting).
  - `Embedding` layer (to represent categorical data in a continuous vector space, commonly used in NLP tasks)
  - `Pooling` layer (to down-sample feature maps and reduce spatial dimensions), i.e. reduce the spatial dimensions of images in CNNs and examine a smaller region of the input at a time.
- **output layer**: final layer that produces the prediction or classification. 
- **forward-propagation**: process of passing input data through the network to get an output.
- **back-propagation**: algorithm for training the network by adjusting weights based on errors.

![NN](./complex-sine-model.png)
<sub>Image by author</sub>

### weights and biases
Weights are the parameters that determine the strength of the connection between neurons. Each connection has an associated weight that influences how much the input from one neuron affects the output of another neuron. During `training`, the neural network **adjusts** these weights to minimize the error in its predictions.
Biases are additional parameters added to the weighted sum of inputs before applying the activation function. They allow the model to shift the activation function, enabling it to better fit the data. Biases help the neural network learn patterns that do not pass through the origin (zero point).

- **Weights** are a specific type of parameter that determine the strength of the connection between two neurons. They scale the input values as they pass from one layer to the next.

- **Parameters** is a broader term. It includes all values that the model learns during training. This means:
  - **Weights** (connections between neurons)
  - **Biases** (constants added to the weighted sum before activation)

**Example:**
For a simple linear neuron: $f(x) = wx + b$
- $w$ is the **weight**
- $b$ is the **bias**
- Both $w$ and $b$ are **parameters**

**Summary Table:**

| Term        | Definition                                      | Example in $f(x) = wx + b$ |
|-------------|-------------------------------------------------|----------------------------|
| Weight      | Strength of input connection                    | $w$                        |
| Parameter   | Any value learned by the model (weights, biases)| $w$, $b$                   |

**Gotcha:**  
All weights are parameters, but not all parameters are weights. Biases are parameters too!

**Trainable vs non-trainable parameters**: Trainable parameters are those that are updated during the training process (e.g., weights and biases), while non-trainable parameters remain fixed (e.g., certain hyperparameters or pre-trained embeddings).

[🧪Demo: simplest network](./neural-network-1.ipynb)

#### activation function
An **activation function** is applied to the output of each neuron after calculating the weighted sum of its inputs and adding the bias. It introduces non-linearity, enabling the neural network to learn complex patterns. 
Without activation functions, a neural network would behave like a simple linear model, regardless of its depth. By applying non-linear activation functions, the network can approximate more complex functions.

**General formula:**  
`output = activation_function(sum(weights * inputs) + bias)`

Common activation functions include Sigmoid, ReLU, and Tanh.
Linear: $f(x) = x$ identity function: it returns the input, unmodified. Typically used in `output layer` for regression tasks
Sigmoid: $σ(x) = 1 / (1 + e^{-x})$ (output values between 0 and 1), typically used in `output layer` for binary classification
Softmax: $σ(z)_i = e^{z_i} / Σ e^{z_j}$ (output values between 0 and 1 that sum to 1), typically used in `output layer` for multi-class classification
ReLU (Rectified Linear Unit): $f(x) = max(0, x)$ (output 0 if x < 0, otherwise output x), typically used in `hidden layers`, particularly in convolutional neural networks (CNNs) and recurrent neural networks (RNNs).
Tanh: $tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})$ (output values between -1 and 1), typically used in `hidden layers` when inputs are centered around zero and data has both positive and negative values.
GELU (Exponential Linear Unit): $f(x) = x$ if $x > 0$, else $α(e^x - 1)$ (output smooth transition for negative inputs), typically used in `hidden layers` to improve learning speed and performance.

[🧪Demo: hidden layers](./neural-network-2.ipynb)

#### loss function and optimizer
Training a neural network involves two key components: the **loss function** and the **optimizer**.

- **loss function**: measures how well the neural network's predictions match the actual target values
    Common loss functions include Mean Squared Error (MSE) for regression tasks and Cross-Entropy Loss for classification tasks. 
    MSE: $(1/n) * Σ(actual - predicted)²$
    Cross-Entropy Loss: $-Σ(actual * log(predicted))$

- **optimizer**: algorithm that adjusts the weights and biases to minimize the loss function during training
    Common optimizers include Stochastic Gradient Descent (SGD) and Adam.
    SGD: updates weights based on the gradient of the loss function with respect to each weight
    Adam: combines the benefits of two other extensions of stochastic gradient descent, specifically Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp)

#### Training key concepts:

- **gradient descent**: iterative optimization algorithm that updates weights in the direction of the negative gradient of the loss function. 
It's a way to slowly adjust its guesses to get better results. It looks at how wrong its current guess is, then changes the guess a little bit to make it less wrong, repeating this process until it finds a good answer.
- **epoch**: one complete pass through the entire training dataset
- **batch size**: number of training examples used in one iteration of gradient descent, for example 32, 64, 128. A smaller batch size provides a regularizing effect and lower memory consumption, while a larger batch size can lead to faster training but may require more memory and can result in poorer generalization.
- **overfitting**: when a model learns the training data too well, including noise, and performs poorly on new data
- **underfitting**: when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and new data
- **learning rate**: hyperparameter that controls the step size during weight updates, for example 0.01, that determines how quickly or slowly a model learns. Slow learning rates may require more epochs to converge, while high learning rates can lead to overshooting the optimal solution. With more complex models, a smaller learning rate is often preferred to ensure stable convergence. With simpler models, a larger learning rate can speed up training without sacrificing performance.

[🧪Demo: image classification](./neural-network-3.ipynb)
[🧪NN Playground](https://playground.tensorflow.org/)

---

## .. and finally contextualizing neural networks & training into LLMs

### LLM Neural Network looks like...

<details>
<img src="./llm-nn.svg" alt="LLM NN" />
<sub>Image generated with Anthropic/Claude</sub>
</details>


---

### LLM training 3-phases
<details>

1. **Pre-training**: The model is trained on a large corpus of text data to learn grammar, facts, and some reasoning abilities. This phase is **unsupervised** and involves predicting the next word in a sentence.
2. **Fine-tuning**: The model is further trained on a smaller, task-specific dataset with human annotations. This phase is **supervised** and helps the model specialize in a particular task, such as question answering or summarization.
3. **Reinforcement Learning with Human Feedback (RLHF)**: The model is refined using feedback from human evaluators. This phase uses **reinforcement learning** to align the model's outputs with human preferences, improving its ability to generate aligned, helpful, and safe responses.
<img src="./3-phases.png" alt="3-phases" />
<sub>Image generated with OpenAI/GPT-5</sub>
</details>
</summary>