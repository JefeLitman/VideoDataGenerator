from DatasetsLoader import VideoDataGenerator
import numpy as np
from sklearn.preprocessing import MinMaxScaler

root_path = "/home/jefelitman/Desktop/VideoDataGenerator/videos_set_1"
batch_size = 2
original_size = [448,300]
size = [112,112]
frames = 20
canales = 3

def custom_temporal_crop(frames):
    mitad = len(frames)//2
    return [frames[mitad - 10: mitad + 10]]

def custom_frame_crop(original_width, original_height):
    return [[0,112,0,112],
            [112,224,112,224]
            ]

def video_transf(video):
    escalador = MinMaxScaler()
    new_video = video.reshape((video.shape[0]*video.shape[1]*video.shape[2]*video.shape[3],1))
    new_video = escalador.fit_transform(new_video)
    return new_video.reshape((video.shape[0],video.shape[1],video.shape[2],video.shape[3]))

def flip_vertical(volume):
    return np.flip(volume, (0, 2))[::-1]

dataset = VideoDataGenerator(directory_path = root_path,
                             table_paths = None,
                             batch_size = batch_size,
                             original_frame_size = original_size,
                             frame_size=size,
                             video_frames = frames,
                             temporal_crop = ("custom", custom_temporal_crop),
                             video_transformation = [("augmented",flip_vertical),("full",video_transf)],
                             frame_crop = ("custom", custom_frame_crop),
                             shuffle = True,
                             conserve_original = False)

print(len(dataset.train_data))