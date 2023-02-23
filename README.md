# DCGAN_generate_sunflower
implementation of dcgan to generate flower

شبکه های مولد متخاصم مشتکل از دو بخش اصلی هستند

۱- Generator

2- Discriminator

در توصیف این دو از api keras استفاده شده است:

 model = tf.keras.Sequential()
 
 ساختار جنراتور به شکل زیر است:
 
 
 noise arraye size = 100
 
 initial shape = (16,16,1024)
 
 layers = 4 * conv2D transpose
 
 
![4](https://user-images.githubusercontent.com/30187615/220946300-cd558d36-9520-4c5a-b2c3-e3e58493a42d.PNG)

همچنین لازم به ذکر است که از تابع فعال سازی RELU برای تمامی لایه ها بجز لایه خروجی استفاده شده است. TANH برای لایه خروجی استفاده شده است

padding = same


ساختار Discriminator:

![5](https://user-images.githubusercontent.com/30187615/220959021-75d26bce-a650-4f70-ad51-d2463db82244.PNG)

layers = 3 * Conv2d
