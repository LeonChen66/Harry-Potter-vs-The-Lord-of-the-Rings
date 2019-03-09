# Harry Potter vs The Lord of the Rings

## A movie classifier using Machine Learning methods

### Introducntio of the project

#### It's always been a great debate: The Lord of the Rings vs. Harry Potter  
#### Which is the greater fantasy movie ? I always vote for Harry Patter, since Harry Potter accompanied my childhood.  

__As a result, I came up with an idea that train models to recognize the input images on Harry Potter & The Lord of the Rings. Moreover, compare different machine learning methods espeically in Computer Vision.__

* __The Lord of the Rings__

<p align="center"> 
    <img src="https://upload.wikimedia.org/wikipedia/en/c/c3/The_Lord_of_the_Rings_trilogy_poster.jpg">
</p>

###### The Lord of the Rings is a film series of three epic fantasy adventure films directed by Peter Jackson, based on the novel The Lord of the Rings by J. R. R. Tolkien. The films are subtitled The Fellowship of the Ring (2001), The Two Towers (2002) and The Return of the King (2003).  
  
###### Set in the fictional world of Middle-earth, the films follow the hobbit Frodo Baggins (Elijah Wood) as he and the Fellowship embark on a quest to destroy the One Ring, to ensure the destruction of its maker, the Dark Lord Sauron. The Fellowship eventually splits up and Frodo continues the quest with his loyal companion Sam (Sean Astin) and the treacherous Gollum (Andy Serkis). Meanwhile, Aragorn (Viggo Mortensen), heir in exile to the throne of Gondor, along with Legolas, Gimli, Merry, Pippin and the wizard Gandalf (Ian McKellen), unite to rally the Free Peoples of Middle-earth in the War of the Ring.

* __Harry Potter__
<p align="center"> 
    <img src="https://www.slso.org/image/production/large/1718_HarryPotter2_large.jpg" style="width:400px;height:400px;">
</p>

###### Harry Potter is a series of fantasy novels written by British author J. K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic, and subjugate all wizards and Muggles (non-magical people).

### Machine Learning Background
##### Over the last years deep learning methods have been shown to outperform previous state-of-the-art machine learning techniques in several fields, with computer vision being one of the most prominent cases. The following 2 figures can expain the difference between conventional Machine Learning and Deep Learning. Deep learning allows computational models of multiple processing layers to learn and represent data with multiple levels of abstraction mimicking how the brain perceives and understands multimodal information, thus implicitly capturing intricate structures of large‐scale data. In computer vision, the most commonly method is convolutional neural network. Finding good internal representations of images objects and features has been the main goal since the beginning of computer vision. Therefore many tools have been invented to deal with images. Many of these are based on a mathematical operation, called convolution. Even when Neural Networks are used to process images, convolution remains the core operation.
  
<p align="center"> 
<img src="https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1542231692/AI_circle_ohnzmy.jpg" style="width:150px;height:150px;">&nbsp;&nbsp;
<img src="https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1542231691/data_bw94vh.png" style="width:250px;height:170px;">
</p>
  

### Experiments Sum up

Dataset : 
* __Harry Potter films : 5007 images__  
The Philosopher's Stone (1997)  
The Chamber of Secrets (1998)  
The Prisoner of Azkaban (1999)  
The Goblet of Fire (2000)  
The Order of the Phoenix (2003)  
The Half-Blood Prince (2005)  
The Deathly Hallows (2007)  

* __The Lord of the Rings films : 2902 images__  
The Fellowship of the Ring (2001)  
The Two Towers (2002)  
The Return of the King (2003)

#### I clip video every 10 sec in the films. 

| Model                             | Accuracy       |
| :--------------------------------:|:--------------:|
| Simple CNN                        | 90.82 %        |
| Transfer Learning with VGG16      |                |
| Random Forest Using HOG           | 73.38 %        |
| Support Vector Machine Using HOG  | 64.22 %        |


#### Let's using simple CNN model to test ! The results look great !
<img src="https://cdn1us.denofgeek.com/sites/denofgeekus/files/styles/main_wide/public/harry_potter_footage_rpg_leak.jpeg?itok=ZDCBGdBt" style="width:250px;height:250px;">

| Harry Potter               | The Lord of the Rings    |
| :-------------------------:|:------------------------:|
| 93.2%                      | 6.8 %                    |
  
  
 
<img src="http://20.theladbiblegroup.com/s3/content/df6a49665a09edb4fe9c6c3738c79f29.jpg" style="width:250px;height:250px;">
  
| Harry Potter               | The Lord of the Rings    |
| :-------------------------:|:------------------------:|
| 0.1%                       | 99.9 %                   |

### Convolutional neural network
##### CNNs use a variation of multilayer perceptrons designed to require minimal preprocessing.They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on their shared-weights architecture and translation invariance characteristics. CNNs use relatively little pre-processing compared to other image classification algorithms. This means that the network learns the filters that in traditional algorithms were hand-engineered. This independence from prior knowledge and human effort in feature design is a major advantage.


＊ A single figure to understand Convolutional neural network  
![alt text](https://cdn-images-1.medium.com/max/1600/1*N4h1SgwbWNmtrRhszM9EJg.png "Logo Title Text 1") 
* The model Architecture
![alt text](https://github.com/LeonChen66/Harry-Potter-vs-The-Lord-of-the-Rings/blob/master/images/cnnmodel.png "Logo Title Text 1")  

* Loss Function of Simple CNN  
![alt text](https://github.com/LeonChen66/Harry-Potter-vs-The-Lord-of-the-Rings/blob/master/images/cnn_loss.png "Logo Title Text 1")

### Transfer Learning with VGG16
* __Schematic Diagram of VGG16 Model:__  
![alt text](https://flyyufelix.github.io/img/vgg16.png "Logo Title Text 1")  
##### Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

##### It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems
<blockquote><p>Transfer learning is the improvement of learning in a new task through the transfer of knowledge from a related task that has already been learned.</p></blockquote>
<p>— <a href="ftp://ftp.cs.wisc.edu/machine-learning/shavlik-group/torrey.handbook09.pdf">Chapter 11: Transfer Learning</a>, <a href="http://amzn.to/2fgeVro">Handbook of Research on Machine Learning Applications</a>, 2009.</p>

### Histogram of oriented gradients(HOG)
##### The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in localized portions of an image. This method is similar to that of edge orientation histograms, scale-invariant feature transform descriptors, and shape contexts, but differs in that it is computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy.
* __HOG feature extraction__  
* 
<img src="https://www.researchgate.net/publication/315808348/figure/fig2/AS:482045051838465@1491939904103/HOG-feature-extraction-a-input-image-b-edge-detection-and-division-into.png" style="width:250px;height:270px;">

### Random Forest
##### Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.
* __Its main idea is "Bagging"__  

![alt text](https://i0.wp.com/upload.wikimedia.org/wikipedia/commons/7/76/Random_forest_diagram_complete.png?zoom=2&w=456&ssl=1 "Logo Title Text 1")  

### Support Vector Machine 
##### Support-vector machines (SVMs, also support-vector networks[1]) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. 
##### More formally, a support-vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection

* __Concept of SVM__    
<img src="https://cdn-images-1.medium.com/max/1600/1*nUpw5agP-Vefm4Uinteq-A.png" style="width:450px;height:250px;">
  
* __Hyperplane__   
  
<img src="https://appliedmachinelearning.files.wordpress.com/2017/03/svm_logo1.png" style="width:250px;height:250px;">
  
__Reference:__   
[1] Voulodimos, A., Doulamis, N., Doulamis, A., & Protopapadakis, E. (2018). Deep learning for computer vision: a brief review. Computational intelligence and neuroscience, 2018.