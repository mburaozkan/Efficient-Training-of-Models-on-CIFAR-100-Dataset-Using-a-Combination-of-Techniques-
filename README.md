# Efficient-Training-of-Models-on-CIFAR-100-Dataset-Using-a-Combination-of-Techniques
This repoistory contains research about Optimization Methods and their implementation to optimize Artificial Neural Network (ANN)

## ABSTRACT
The performance and application of several algorithms are the main topics of this report's investigation into optimization techniques for neural networks. We examine the widely used gradient-based optimization techniques of stochastic gradient descent and batch gradient descent. In addition, we look at gradient-free methods like the genetic algorithm, particle swarm optimization, and simulated annealing. We examine the benefits and drawbacks of each approach, stressing both. We also investigate the advantages of mixing gradient-free and gradient-based techniques for neural network optimization. The research paves the ground for additional developments in neural network optimization by offering insights into the strengths and weaknesses of these methods.
  
 ## LITERATURE SURVEY
 Gradient Descent (GD) and Stochastic Gradient Descent (SGD) are searched gradient-based optimization methods for training neural networks. GD updates the model parameters by computing the gradients of the loss function with respect to the parameters and moving in the direction of steepest descent. SGD performs parameter updates using a randomly selected subset (mini-batch) of training examples, which offers computational efficiency. Bottou et. al. outline a thorough theory of a simple yet adaptable SGD algorithm, go over its real-world behavior, and identify areas where building algorithms with better performance is possible. These methods have been the foundation for training neural networks and serve as the baseline for evaluating the effectiveness of alternative optimization approaches.

 Particle Swarm Optimization (PSO), Simulated Annealing (SA), and Genetic Algorithm (GA) are examples of gradient-free optimization methods that have been studied to training neural networks. PSO, proposed by Kennedy and Eberhart, is inspired by the collective behavior of bird flocks or fish schools and uses a swarm of particles to explore the parameter space. SA, introduced by Kirkpatrick et al., simulates the annealing process in metallurgy, gradually cooling the system to escape local optima. GA, developed by Holland, is a population-based optimization algorithm inspired by the principles of natural selection and genetics. These gradient-free methods offer different strategies for exploring the parameter space and have shown promising results in improving neural network optimization.

  Several studies have investigated the combination of gradient-based and gradient-free optimization methods for neural network training. Li et al. conducted a comprehensive survey on hybrid optimization algorithms and discussed the advantages and challenges of integrating gradient-free methods with gradient-based methods [4]. They explored the combination of PSO or GA with GD or SGD, highlighting the potential benefits in terms of improved convergence and robustness. Other researchers have also investigated hybrid approaches that combine SA with gradient-based methods or used a combination of genetic operators with GD or SGD [7, 8]. These studies have demonstrated the potential of combining both types of optimization methods to enhance neural network training.

  
## Environment
 - Windows 11
 - Python 3.9
 - matplotlib	3.3
 - numpy	1.19.5
 - scikit-learn	0.24
 - scipy	1.6.0	

## Demo and The Efficient of Results
the following diagram and reports shows the performance of testing data of the dataset including 10 classes (digits classes)

<p align="center" width="100%">
  <img src="https://res.cloudinary.com/dvdcninhs/image/upload/v1610484745/testingPSO_eejk9w.png" width="500" hight="500"/>
</p>
<p align="center" width="100%">
  <img src="https://res.cloudinary.com/dvdcninhs/image/upload/v1610484745/plotPSO_oiqreo.png" width="500" hight="500"/>
</p>

## Conclusion

In conclusion, this study emphasizes the value of optimization techniques in raising the effectiveness of models. It is urged that further work be done in this area to investigate novel approaches, adapt current methodologies to particular applications, and handle the difficulties posed by massive datasets and intricate model structures. We can fully utilize the capabilities of machine learning models by developing optimization strategies, which will result in more precise forecasts, better decision-making, and greater performance in real-world scenarios.
