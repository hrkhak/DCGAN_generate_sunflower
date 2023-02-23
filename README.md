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

---------------------------------------------------------------------------------------------

دیتاست از ادرس 
 http://download.tensorflow.org/example_images/flower_photos.tgz
 گرفته شده است
 که شامل پوشه گل های رز و افتاب گردان و ... می باشد.
 
 
 ![1](https://user-images.githubusercontent.com/30187615/220960508-1fe96834-6bbe-40a1-812a-306734541ee2.PNG)




---------------------------------------------------------------------------------------------

*داده افزایی*
*Data Augmentation*


به جهت داده افزایی اقداماتی نظیر چرخش تصادفی - وارونه نمایی عمودی و افقی و برش صورت پذیرفته است.




available_transformations = {

    'rotate': random_rotation,
    
    'vertical_flip': vertical_flip,
    
    'horizontal_flip': horizontal_flip,
    
    'vertical_and_horizontal_flip': vertical_and_horizontal_flip,
    
    'TF_crop_pad': TF_crop_pad
    
}
 
* نمونه داده افزایی بر روی یک تصویر
 
 
 
![horizental flip](https://user-images.githubusercontent.com/30187615/221011697-d37a9c4e-17bd-42a3-821f-ce0eb9b09556.PNG)
![random rotation](https://user-images.githubusercontent.com/30187615/221011716-390eab92-24c0-4ec1-a18a-d5c88b4063c5.PNG)
![vertical flip](https://user-images.githubusercontent.com/30187615/221011733-f3f64afd-7445-454f-93cf-324120caf772.PNG)
![vertical and horizental flip](https://user-images.githubusercontent.com/30187615/221011750-dbb03c7c-1558-4521-976d-dd9227862fa8.PNG)


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
