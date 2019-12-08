from DatasetsLoader import VideoDataGenerator

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

dataset = VideoDataGenerator(directory_path = root_path,
                             batch_size = batch_size,
                             original_frame_size = original_size,
                             frame_size=size,
                             video_frames = frames,
                             temporal_crop = ("custom", custom_temporal_crop),
                             frame_crop = ("custom", custom_frame_crop),
                             shuffle = False,
                             conserve_original = False)

print(len(dataset.train_data))