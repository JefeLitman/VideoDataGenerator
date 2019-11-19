# <u>VideoDataGenerator</u>: A easy data tool for machine learning with videos
This option is similar to the keras.ImageDataGenerator how take the data
in a folder and load sequentially from it.

**Important:** For the moment the VideoDataGenerator works only with 
the notation of channels in the last dimension. NFHWC (N - Batch, F - Frames, 
H - Height, W - Widht and C - Channels). 

**Actual version: v1.7 (If you ask why the jump from v1.0 to v1.7, the reason 
is that the older version are in my repository 
[Biv2LabNN](https://github.com/JefeLitman/Biv2LabNN/commits/master))**

- Features:
    - The option of `original_size` that specified the original size of the frames 
    to work.
    - Change the behaviour of the `conserve_original` parameter when it is
    specified to True.
    
- Future features:
    - It will work with the API of tensorflow `Dataset.from_generator`.
    - Add the `frame_trasnformation`and `video_transformation`.
    - Do all the heavy task in parallel with threads and multiprocessors.
    - Do the `frame_trasnformation`and `video_transformation` in parallel 
    with threads and multiprocessors.

# Documentation

### Installation
Just copy the file in your project and import the class..

`from DatasetsLoader import VideoDataGenerator`

### Dependecies
This file have only two dependecies, the opencv library (Only use imread, 
cvtColor and resize) and the numpy library. It doesn't matter the version so relax and install 
whatever you want ;)

### How to use it?
Well it's simple, first we must understand how the directories must be in 
order to `VideoDataGenerator` works:

- Dataset_directory
    - train
        - Classes (In folder)
            - Videos (In folders)
                - Frames (Files in jpg, png, tiff or ppm)
    - test
        - Classes (In folder)
            - Videos (In folders)
                - Frames (Files in jpg, png, tiff or ppm)
    - dev
        - Classes (In folder)
            - Videos (In folders)
                - Frames (Files in jpg, png, tiff or ppm)
                
If you see, yes... Only accepts the folders of train, test and dev data (Dev is 
optional but train and test are required) so order you dataset and enjoy 
this tool for your projects.

**In a future it will implement a method to read the avi files to avoid the 
process of spliting the frames of the videos**

When you create the `VideoDataGenerator` it will ask your for this parameters:
- `directory_path`: String of the dataset path. **Obligatory**.
- `batch_size`: Default in 32, it specifies the size of batches to generate.
- `original_frame_size`: Default None, it resize the original image before
applying a transformation over it. None means the original size and you must pass the 
size in a tuple like `(width, height)`.
- `frame_size`: Default None, it specifies the final image size to return
after applying transformations. None means the original size and you must pass the
size in a tuple like `(width, height)`.
- `video_frames`: Default 16, it specifies the final video frames to return.
- `temporal_crop`: Default is `(None, None)`, it specifies what type of operation 
over the temporal axis must be done. For more information read the below section.
- `video_transformation`: Default None, it specifies what transformation 
must be done over the video after loaded. For more information read the below section.
**Future release**
- `frame_crop`: Default is (None, None), it specifies what type of operation 
over the spatial axis must be done. For more information read the below section.
- `frame_transformation`: Default None, it specifies what transformation 
must be done in every frame of a video after loaded. For more information 
read the below section. **Future release**
- `shuffle`: Default False, Boolean that specifies if the data must be shuffle 
or not.
- `conserve_original`: Default False, Boolean that specifies if for every 
transformation done in the data the original form of the data should be
conserved. For more information read the below section.

### Transformation and basics

In construction

### Do you want to contribute?
- **Core explanation**

In construction

- **Data structure explanation**

In construction

- **How it load the data**

In construction