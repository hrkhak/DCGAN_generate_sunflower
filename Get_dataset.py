import tarfile
GET_DATASET = True
data_dir = '/content/flower_photos'
#DATASET_FOLDER = 'content/flower_photos/'
if GET_DATASET:
    print('Fechting the Image Data Set')
    !wget http://download.tensorflow.org/example_images/flower_photos.tgz
    print('Unzipping the file')
    ZIP_FILE = '/content/flower_photos.tgz'
    #!rm -rf only_flowers/
    #!unzip -q -o $ZIP_FILE
    tarfile.open(ZIP_FILE, 'r:gz').extractall(data_dir)
    print('Daisies images are available')
#Check if there are some files in the folder
num_of_images = len(os.listdir(DATASET_FOLDER))
print('Number of images: ', num_of_images)
