# Movie-Classifier

## A movie classifier using Machine Learning methods

### Introducntio of the project

It's always been a great debate: The Lord of the Rings vs. Harry Potter  
Which is the greater fantasy movie ? I always vote for Harry Patter, since Harry Potter accompanied my childhood.  

__As a result, I came up with an idea that train models to recognize the input images on Harry Potter & The Lord of the Rings. Moreover, compare different machine learning methods espeically in Computer Vision.__

* __The Lord of the Rings__

<p align="center"> 
    <img src="https://upload.wikimedia.org/wikipedia/en/c/c3/The_Lord_of_the_Rings_trilogy_poster.jpg">
</p>
The Lord of the Rings is a film series of three epic fantasy adventure films directed by Peter Jackson, based on the novel The Lord of the Rings by J. R. R. Tolkien. The films are subtitled The Fellowship of the Ring (2001), The Two Towers (2002) and The Return of the King (2003).  
  
Set in the fictional world of Middle-earth, the films follow the hobbit Frodo Baggins (Elijah Wood) as he and the Fellowship embark on a quest to destroy the One Ring, to ensure the destruction of its maker, the Dark Lord Sauron. The Fellowship eventually splits up and Frodo continues the quest with his loyal companion Sam (Sean Astin) and the treacherous Gollum (Andy Serkis). Meanwhile, Aragorn (Viggo Mortensen), heir in exile to the throne of Gondor, along with Legolas, Gimli, Merry, Pippin and the wizard Gandalf (Ian McKellen), unite to rally the Free Peoples of Middle-earth in the War of the Ring.

* __Harry Potter__
<p align="center"> 
    <img src="https://www.slso.org/image/production/large/1718_HarryPotter2_large.jpg" style="width:400px;height:400px;">
</p>
Harry Potter is a series of fantasy novels written by British author J. K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic, and subjugate all wizards and Muggles (non-magical people).

### Machine Learning Background
Over the last years deep learning methods have been shown to outperform previous state-of-the-art machine learning techniques in several fields, with computer vision being one of the most prominent cases. The following 2 figures can expain the difference between conventional Machine Learning and Deep Learning. Deep learning allows computational models of multiple processing layers to learn and represent data with multiple levels of abstraction mimicking how the brain perceives and understands multimodal information, thus implicitly capturing intricate structures of large‐scale data. In computer vision, the most commonly method is convolutional neural network.
  
<p align="center"> 
<img src="https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1542231692/AI_circle_ohnzmy.jpg" style="width:150px;height:150px;">&nbsp;&nbsp;
<img src="https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1542231691/data_bw94vh.png " style="width:250px;height:170px;">
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

###### I clip video every 10 sec in the films. 

| Model                             | Accuracy       |
| :--------------------------------:|:--------------:|
| Simple CNN                        | 90.82 %        |
| Transfer Learning with VGG16      |                |
| Random Forest Using Hog           | 73.38 %        |
| Support Vector Machine Using Hog  | 64.22 %        |


###### Let's using simple CNN model to test ! The results look great !
<p align="center"> 
    <img src="https://cdn1us.denofgeek.com/sites/denofgeekus/files/styles/main_wide/public/harry_potter_footage_rpg_leak.jpeg?itok=ZDCBGdBt" style="width:400px;height:400px;">
</p>
| Harry Potter               | The Lord of the Rings    |
| :-------------------------:|:------------------------:|
| 93.2%                      | 6.8 %                    |
  
  
<p align="center"> 
    <img src="http://20.theladbiblegroup.com/s3/content/df6a49665a09edb4fe9c6c3738c79f29.jpg" style="width:400px;height:400px;">
</p>
| Harry Potter               | The Lord of the Rings    |
| :-------------------------:|:------------------------:|
| 0.1%                       | 99.9 %                    |


Ref: Voulodimos, A., Doulamis, N., Doulamis, A., & Protopapadakis, E. (2018). Deep learning for computer vision: a brief review. Computational intelligence and neuroscience, 2018.