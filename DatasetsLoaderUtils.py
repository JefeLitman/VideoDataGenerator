"""VideoDatasetsLoader file for any Video Data to be loaded with a generator.
The requieriments are Numpy, opencv and pandas.
"""

import os
import cv2
import numpy as np
import pandas as pd
import types

class load_videoFrames_from_path():
    """Class to take frames of videos from a folder structure and returns
    the generators to train, test and optionally dev.
    Version 1.4
    """
    def __init__(self,
                 directory_path,
                 video_frames,
                 frames_size = None,
                 data_format = 'channels_last'
                 ):
        """Initializer
        Args:
            directory_path: String that constains the dataset path splitted in train, test and 
                            maybe dev. Obligatory in case table_paths is not used.
            video_frames: Python callback that receives only the video and returns the video with the temporal
                          axis as the user want.
            frames_size: Integer tuple with the final frame size for videos in the order (width, height).
                        Default in None and will take the original size of frames in video.
            data_format: String, one of 'channels_last' or 'channels_first'. Default in 'channels_last'.
            """
        self.__set_parameters__(directory_path, video_frames, frames_size, data_format)

        self.__build_data_folders__(directory_path)

        self.__generate_classes__()

        self.__get_video_paths__()

    def __set_parameters__(self, directory_path, video_frames, frames_size, data_format):
        # Check parameters types
        if not isinstance(directory_path, str):
            raise TypeError('Directory_path must be a string. Type given: '+str(type(directory_path)))
        elif not isinstance(video_frames, types.FunctionType):
            raise TypeError('Video_frames must be a python callback. Type given: '+str(type(video_frames)))
        elif not isinstance(frames_size, (tuple, list, type(None))):
            raise TypeError('Frames_size can be a Tuple or List of integers. Type given: '+str(type(frames_size)))
        elif data_format not in ('channels_last', 'channels_first'):
            raise TypeError('Data_format can be "channels_last" or "channels_first". Value given: '+str(data_format))

        # Set basic attributes
        self.__transformation__ = video_frames
        self.__size__ = frames_size
        self.__data_format__ = data_format

    def __build_data_folders__(self, directory_path):
        self.__dev_path__ = None
        directories = os.listdir(directory_path)
        for i in directories:
            if i.lower() == "train":
                self.__train_path__ = os.path.join(directory_path, i)
            elif i.lower() == "test":
                self.__test_path__ = os.path.join(directory_path, i)
            elif i.lower() == "dev":
                self.__dev_path__ = os.path.join(directory_path, i)
            else:
                raise ValueError(
                    'The folder must have the structure of train, test and dev '
                    '(dev is optional) with the same name case insensivity. '
                    'Folder of the problem: %s' % i)
    
    def __generate_classes__(self):
        self.to_class = [clase.lower() for clase in sorted(os.listdir(self.__train_path__))]
        self.to_number = dict((name, index) for index, name in enumerate(self.to_class))

    def __get_video_paths__(self):
        self.__videos_train_path__ = []
        self.__videos_test_path__ = []
        if self.__dev_path__:
            self.__videos_dev_path__ = []

        for clase in sorted(os.listdir(self.__train_path__)):
            videos_train_path = os.path.join(self.__train_path__,clase)
            self.__videos_train_path__ += [os.path.join(videos_train_path,i) for i in sorted(os.listdir(videos_train_path))]

            videos_test_path = os.path.join(self.__test_path__,clase)
            self.__videos_test_path__ += [os.path.join(videos_test_path,i) for i in sorted(os.listdir(videos_test_path))]

            if self.__dev_path__:
                videos_dev_path = os.path.join(self.__dev_path__,clase)
                self.__videos_dev_path__ += [os.path.join(videos_dev_path,i) for i in sorted(os.listdir(videos_dev_path))]

    def __load_video__(self, video_path, channels = 3):
        video = []
        frames_path = [os.path.join(video_path, frame) for frame in sorted(os.listdir(video_path))]

        for frame in frames_path:
            image = self.__load_frame__(frame, channels)
            video.append(image)

        video = np.asarray(video, dtype=np.float32)
        if channels == 1:
            video = video.reshape((video.shape[0], video.shape[1], video.shape[2], 1))
        if self.__data_format__ == 'channels_last':
            return video
        else:
            return np.moveaxis(video, 3, 0)

    def __load_frame__(self, frame_path, channels = 3):
        if channels == 1:
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        elif channels == 3:
            img =  cv2.cvtColor(cv2.imread(frame_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

        if self.__size__:
            return cv2.resize(img, tuple(self.__size__))
        else:
            return img

    def data_generator(self, data_type, channels):
        if data_type == 1:
            data = self.__videos_train_path__
        elif data_type == 2:
            data = self.__videos_test_path__
        elif data_type == 3:
            if self.__dev_path__:
                data = self.__videos_dev_path__
            else:
                raise NotImplementedError(
                    'You can not get the dev generator because there is no data in it.'
                )
        else:
            raise ValueError(
                'The data_type given is not 1 to "train", 2 to "test" or 3 to "dev". data_type given {i}'.format(i=data_type)
            )
        for video_path in data:
            label = self.to_number[video_path.split("/")[-2].lower()]
            video = self.__load_video__(video_path, channels)
            video = self.__transformation__(video)
            if self.__size__:
                if self.__data_format__ == 'channels_last':
                    if video.shape[1:] != (self.__size__[1], self.__size__[0], channels):
                        raise AssertionError(
                            'The python callback given in video_transformation returns an invalid '
                            'video shape. Video dimension return: '+str(video.shape)+' \nDimension '
                            'that must be return: ' + str((video.shape[0], self.__size__[1], self.__size__[0], channels))
                        )
                else:
                    if video.shape[0] == channels and video.shape[2] == self.__size__[1] and video.shape[3] == self.__size__[0]:
                        raise AssertionError(
                            'The python callback given in video_transformation returns an invalid '
                            'video shape. Video dimension return: '+str(video.shape)+' \nDimension '
                            'that must be return: ' + str((channels, video.shape[1], self.__size__[1], self.__size__[0]))
                        )
            yield video, label

class load_videoFiles_from_path():
    """Class to take video files from a folder structure and returns
    the generators to train, test and optionally dev.
    Version 0.1
    """
    def __init__(self):
        raise NotImplementedError('This util is not implemented yet... Sorry :(')

class flow_from_tablePaths():
    """Class to take a dataframe, numpy matrix or Python list of lists 
    to read video data and returns the generators to train, test and 
    optionally dev.
    Version 1.4
    """
    def __init__(self,
                 table_paths,
                 video_frames,
                 frames_size = None,
                 data_format = 'channels_last'
                 ):
        """Initializer
        Args:
            table_paths: Pandas Dataframe, numpy array or python matrix with shape (n_videos, 3)
                         where the columns are: (video_path, video_type, label) and video type
                         can be "train", "test" or "dev" only (In a string format and case insentivity).
            video_frames: Python callback that receives only the video and returns the video with the temporal
                          axis as the user want.
            frames_size: Integer tuple with the final frame size for videos in the order (width, height).
                        Default in None and will take the original size of frames in video.
            data_format: String, one of 'channels_last' or 'channels_first'. Default in 'channels_last'.
            """
        self.__set_parameters__(table_paths, video_frames, frames_size, data_format)

        self.__build_data__()

        self.__generate_classes__()

    def __set_parameters__(self, table_paths, video_frames, frames_size, data_format):
        # Check parameters types
        if not isinstance(table_paths, (list, np.ndarray, pd.DataFrame)):
            raise TypeError('Table_paths must be a dataframe, numpy array or python list of lists. Type given: '+str(type(table_paths)))
        elif not isinstance(video_frames, types.FunctionType):
            raise TypeError('Video_frames must be a python callback. Type given: '+str(type(video_frames)))
        elif not isinstance(frames_size, (tuple, list, type(None))):
            raise TypeError('Frames_size can be a Tuple or List of integers. Type given: '+str(type(frames_size)))
        elif data_format not in ('channels_last', 'channels_first'):
            raise TypeError('Data_format can be "channels_last" or "channels_first". Value given: '+str(data_format))

        # Set basic attributes
        if isinstance(table_paths, list):
            self.__data__ = np.r_[table_paths]
        elif isinstance(table_paths, pd.DataFrame):
            self.__data__ = table_paths.values
        else:
            self.__data__ = table_paths
        self.__transformation__ = video_frames
        self.__size__ = frames_size
        self.__data_format__ = data_format

    def __build_data__(self):
        self.__videos_train_path__ = []        
        self.__train_indexes__ = []
        self.__videos_test_path__ = []
        self.__test_indexes__ = []
        videos_dev_path = []
        dev_indexes = []

        for index, video_param in enumerate(self.__data__):
            if str(video_param[1]).lower() == "train":
                self.__videos_train_path__ += [str(video_param[0])]
                self.__train_indexes__ += [index]
            elif str(video_param[1]).lower() == "test":
                self.__videos_test_path__ += [str(video_param[0])]
                self.__test_indexes__ += [index]
            elif str(video_param[1]).lower() == "dev":
                videos_dev_path += [str(video_param[0])]
                dev_indexes += [index]
            else:
                raise AssertionError(
                    'Inside table_paths exists a video_type invalid, The valid values are '
                    '"train", "test" and "dev". Value given: ' + str(video_param[1]))

        if len(videos_dev_path) > 0:
            self.__dev_indexes__ = dev_indexes
            self.__videos_dev_path__ = videos_dev_path
        else:
            self.__dev_indexes__ = False
    
    def __generate_classes__(self):
        self.to_class = [str(clase).lower() for clase in np.unique(self.__data__[:,2])]
        self.to_number = dict((name, index) for index, name in enumerate(self.to_class))

    def __load_video__(self, video_path, channels = 3):
        video = []
        frames_path = [os.path.join(video_path, frame) for frame in sorted(os.listdir(video_path))]

        for frame in frames_path:
            image = self.__load_frame__(frame, channels)
            video.append(image)

        video = np.asarray(video, dtype=np.float32)
        if channels == 1:
            video = video.reshape((video.shape[0], video.shape[1], video.shape[2], 1))
        if self.__data_format__ == 'channels_last':
            return video
        else:
            return np.moveaxis(video, 3, 0)

    def __load_frame__(self, frame_path, channels = 3):
        if channels == 1:
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        elif channels == 3:
            img =  cv2.cvtColor(cv2.imread(frame_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

        if self.__size__:
            return cv2.resize(img, tuple(self.__size__))
        else:
            return img

    def data_generator(self, data_type, channels):
        if data_type == 1:
            data = self.__videos_train_path__
            indexes = self.__train_indexes__
        elif data_type == 2:
            data = self.__videos_test_path__
            indexes = self.__test_indexes__
        elif data_type == 3:
            if self.__dev_indexes__:
                data = self.__videos_dev_path__
                indexes = self.__dev_indexes__
            else:
                raise NotImplementedError(
                    'You can not get the dev generator because there is no data in it.'
                )
        else:
            raise ValueError(
                'The data_type given is not 1 to "train", 2 to "test" or 3 to "dev". data_type given {i}'.format(i=data_type)
            )
        for index, video_path in enumerate(data):
            label = self.to_number[str(self.__data__[indexes[index], 2]).lower()]
            video = self.__load_video__(video_path, channels)
            video = self.__transformation__(video)
            if self.__size__:
                if self.__data_format__ == 'channels_last':
                    if video.shape[1:] != (self.__size__[1], self.__size__[0], channels):
                        raise AssertionError(
                            'The python callback given in video_transformation returns an invalid '
                            'video shape. Video dimension return: '+str(video.shape)+' \nDimension '
                            'that must be return: ' + str((video.shape[0], self.__size__[1], self.__size__[0], channels))
                        )
                else:
                    if video.shape[0] == channels and video.shape[2] == self.__size__[1] and video.shape[3] == self.__size__[0]:
                        raise AssertionError(
                            'The python callback given in video_transformation returns an invalid '
                            'video shape. Video dimension return: '+str(video.shape)+' \nDimension '
                            'that must be return: ' + str((channels, video.shape[1], self.__size__[1], self.__size__[0]))
                        )
            yield video, label

class VideoDataGenerator():
    """Class to load all data from a Video Dataset with paths or Dataframe
    specified by user and add easy transformations executed in real time.
    Only take data from frames files and not from video files.
    Version 2.2.8
    """

    def __init__(self,
                 directory_path = None,
                 table_paths = None,
                 batch_size = None,
                 original_frame_size = None,
                 frame_size = None,
                 video_frames = None,
                 temporal_crop = (None, None),
                 video_transformation = None,
                 frame_crop = (None, None),
                 shuffle = False,
                 shuffle_after_epoch = False,
                 conserve_original = False
                 ):
        """Initializer
        Args:
            directory_path: String that constains the dataset path splitted in train, test and 
                            maybe dev. Obligatory in case table_paths is not used.
            table_paths: Pandas Dataframe, numpy array or python matrix with shape (n_videos, 3)
                         where the columns are: (video_path, video_type, label) and video type
                         can be "train", "test" or "dev" only (In a string format). Obligatory in 
                         case directory_path is not used.
            batch_size: Integer that represents the elements for batch. When is None will take the 
                        less quantity of data in all splits or 32 if the data is bigger than 32.
                        Default in None.
            original_frame_size: Integer tuple of original size of frame in the order (width, height).
                                 Default in None and will take the original size of frames in video.
            frame_size: Integer tuple with the final frame size for videos in the order (width, height).
                        Default in None and will take the original size of frames in video.
            video_frames: Integer that specifies the frame number of videos. Default in None and will take
                          the video with less frames in the splits.
            temporal_crop: Tuple in the order (mode, parameter) when the parameter depends of the mode selected.
                           This attribrute refers at how to operate the temporal axis in videos. Default in (None, None).
            video_transformation: Python list of tuples in the order (aplicability, python callback). The 
                                  aplicability can be "full" or "augmented". The order of list is the order
                                  to apply the transformations. By default in None.
            frame_crop: Tuple in the order (mode, parameter) when the parameter depends of the mode selected.
                        This attribrute refers at how to operate the spatial axis in videos. Default in (None, None).
            shuffle: Boolean that shuffle the data in the beginning. Default in False.
            shuffle_after_epoch: Boolean that shuffle the data after complete an epoch. Default in False.
            conserve_original: Boolean that save the original data with the transformated data. Deprecated.
                               Default in False.

            It works with the notation 'channels_last'"""

        """Attributes, restrictions and constants definition"""
        temporal_crop_modes = (None,'sequential','random','custom')
        frame_crop_modes = (None,'sequential','random','custom')

        method_flag = 0
        self.transformation_index = 0
        self.dev_path = None
        self.dev_data = None
        self.shuffle = shuffle_after_epoch

        """Check the path have the establish hierarchy if a path is used"""
        if directory_path is not None:
            method_flag = 1
            self.using_folders(directory_path)

        """Check table_paths is well structured"""
        if table_paths is not None:
            if method_flag == 1:
                raise ValueError('You should use one of the methods to load data but not both. '
                                 'Methods used: {dp} with {df}.'.format(dp=directory_path, df=table_paths))
            method_flag = 2
            self.using_table_paths(table_paths)

        if method_flag == 0:
            raise ValueError('At least use one of the methods. Both parameters were {a}.'.format(a=None))

        """Check temporal and frame crop selected"""
        if temporal_crop[0] not in temporal_crop_modes:
            raise ValueError(
                'The only modes availables to use in temporal crop are '
                '(None, "sequential", "random", "custom"). The selected mode was '
                '%s' % temporal_crop[0]
            )
        if frame_crop[0] not in frame_crop_modes:
            raise ValueError(
                'The only modes availables to use in frame crop are '
                '(None, "sequential", "random", "custom"). The selected mode was '
                '%s' % frame_crop[0]
            )

        """Generation of video paths"""
        self.generate_classes(method_flag)
        self.generate_videos_paths(method_flag)

        """Definition of batch_size"""
        if batch_size:
            self.batch_size = batch_size
        else:
            minimo = 32
            if len(self.videos_train_path) < minimo:
                minimo = len(self.videos_train_path)
            elif len(self.videos_test_path) < minimo:
                minimo = len(self.videos_test_path)
            else:
                if self.dev_path:
                    minimo = len(self.videos_dev_path)
            self.batch_size = minimo

        """Definition of original_size and frame_size"""
        if original_frame_size:
            self.original_size = original_frame_size
        else:
            frames_path = os.path.join(self.videos_train_path[0], sorted(os.listdir(self.videos_train_path[0]))[0])
            self.original_size = self.load_raw_frame(frames_path, original_size_created=False).shape[1::-1]
        if frame_size:
            self.frame_size = frame_size
        else:
            self.frame_size = self.original_size

        """Check the frame sizes are well defined"""
        if frame_crop[0] == 'sequential' and conserve_original == False:
            if min([self.original_size[0] // self.frame_size[0], self.original_size[1] // self.frame_size[1]]) == 0:
                raise ValueError(
                    'Is not possible to load the data with the sizes specified due to the '
                    'final size ({w}, {h}) is bigger than the original size.'.format(w = self.frame_size[0],
                                                                                h = self.frame_size[1])
                    )

        """Definition of video_frames"""
        if video_frames:
            self.video_frames = video_frames
        else:
            minimo = len(os.listdir(self.videos_train_path[0]))
            for video in self.videos_train_path:
                nro_frames = len(os.listdir(video))
                if nro_frames < minimo:
                    minimo = nro_frames
            for video in self.videos_test_path:
                nro_frames = len(os.listdir(video))
                if nro_frames < minimo:
                    minimo = nro_frames
            if self.dev_path:
                for video in self.videos_dev_path:
                    nro_frames = len(os.listdir(video))
                    if nro_frames < minimo:
                        minimo = nro_frames
            self.video_frames = minimo

        """Generate the data with or without transformations"""
        if conserve_original and temporal_crop[0] not in (None, 'sequential'):
            self.temporal_crop(mode = 'sequential', custom_fn=temporal_crop[1], method_flag=method_flag)
        self.temporal_crop(mode = temporal_crop[0], custom_fn=temporal_crop[1], method_flag=method_flag)
        self.frame_crop(mode=frame_crop[0], custom_fn=frame_crop[1], conserve_original=conserve_original)

        """Complete the batches"""
        self.complete_batches()

        """Check transformations"""
        if video_transformation:
            self.trans_train_indexes = []
            self.trans_test_indexes = []
            if self.dev_path:
                self.trans_dev_indexes = []
            try:
                len_train = len(self.train_data)
                len_test = len(self.test_data)
                if self.dev_path:
                    len_dev = len(self.dev_data)
                for mode, _ in video_transformation:
                    if mode.lower() == "augmented":
                        self.trans_train_indexes.append(len_train // self.batch_size)
                        self.train_batches *= 2
                        len_train *= 2

                        self.trans_test_indexes.append(len_test // self.batch_size)
                        self.test_batches *= 2
                        len_test *= 2
                        if self.dev_path:
                            self.trans_dev_indexes.append(len_dev // self.batch_size)
                            self.dev_batches *= 2
                            len_dev *= 2
                    elif mode.lower() == "full":
                        self.trans_train_indexes.append(0)
                        self.trans_test_indexes.append(0)
                        if self.dev_path:
                            self.trans_dev_indexes.append(0)
                    else:
                        raise AttributeError(
                            'Exist a invalid value in transformations. The aplicabilities are '
                            '"full" or "augmented". Value given: '+str(mode))
            except:
                raise ValueError(
                    'Video_transformation must be a python list of tuples with the order (aplicability, '
                    'python callback) where aplicability can be "full" or "augmented". Value given '
                    +str(video_transformation))
            self.video_transformation = [pair[1] for pair in video_transformation]
        else:
            self.video_transformation = video_transformation

        """Shuffle data"""
        if shuffle:
            self.shuffle_videos()

    def using_folders(self, directory_path):
        directories = os.listdir(directory_path)
        for i in directories:
            if i.lower() == "train":
                self.train_path = os.path.join(directory_path, i)
                self.train_batch_index = 0
                self.train_data = []
            elif i.lower() == "test":
                self.test_path = os.path.join(directory_path, i)
                self.test_batch_index = 0
                self.test_data = []
            elif i.lower() == "dev":
                self.dev_path = os.path.join(directory_path, i)
                self.dev_batch_index = 0
                self.dev_data = []
            else:
                raise ValueError(
                    'The folder must have the structure of train, test and dev '
                    '(dev is optional) with the same name case insensivity. '
                    'Folder of the problem: %s' % i)

    def using_table_paths(self, table_paths):
        if isinstance(table_paths, list):
            self.df = np.r_[table_paths]
        elif isinstance(table_paths, pd.DataFrame):
            self.df = table_paths.values
        elif isinstance(table_paths, np.ndarray):
            self.df = table_paths
        else:
            raise ValueError(
                'Table_paths must be a dataframe, numpy array or python list of lists. '
                'Type given: ' + str(type(table_paths)))

        if self.df.shape[1] != 3 and self.df.ndim != 2:
            raise ValueError(
                'Table_paths is not a matrix or have less or more than 3 columns. Dimensions given: '
                + str(self.df.shape))

    def generate_classes(self, method_flag):
        if method_flag == 1:
            self.to_class = [clase.lower() for clase in sorted(os.listdir(self.train_path))] #Equivale al vector de clases
        elif method_flag == 2:
            self.to_class = [str(clase).lower() for clase in np.unique(self.df[:,2])]
        else:
            raise ValueError(
                'The method_flag to generate_classes must be an integer between 1 or 2. Value given: '
                +str(method_flag))
        self.to_number = dict((name, index) for index, name in enumerate(self.to_class))

    def generate_videos_paths(self, method_flag):
        self.videos_train_path = []
        self.videos_test_path = []
        if self.dev_path:
            self.videos_dev_path = []
        if method_flag == 1:
            for clase in sorted(os.listdir(self.train_path)):

                videos_train_path = os.path.join(self.train_path,clase)
                self.videos_train_path += [os.path.join(videos_train_path,i) for i in sorted(os.listdir(videos_train_path))]

                videos_test_path = os.path.join(self.test_path,clase)
                self.videos_test_path += [os.path.join(videos_test_path,i) for i in sorted(os.listdir(videos_test_path))]

                if self.dev_path:
                    videos_dev_path = os.path.join(self.dev_path,clase)
                    self.videos_dev_path += [os.path.join(videos_dev_path,i) for i in sorted(os.listdir(videos_dev_path))]
        elif method_flag == 2:
            videos_dev_path = []
            dev_indexes = []
            self.train_indexes = []
            self.test_indexes = []

            for index, video_param in enumerate(self.df):
                if str(video_param[1]).lower() == "train":
                    self.videos_train_path += [str(video_param[0])]
                    self.train_indexes += [index]
                elif str(video_param[1]).lower() == "test":
                    self.videos_test_path += [str(video_param[0])]
                    self.test_indexes += [index]
                elif str(video_param[1]).lower() == "dev":
                    videos_dev_path += [str(video_param[0])]
                    dev_indexes += [index]
                else:
                    raise AssertionError(
                        'Inside table_paths exists a video_type invalid, The valid values are '
                        '"train", "test" and "dev". Value given: ' + str(video_param[1]))

            self.train_data = []
            self.train_batch_index = 0

            self.test_data = []
            self.test_batch_index = 0

            if len(videos_dev_path) > 0:
                self.dev_path = True
                self.dev_indexes = dev_indexes
                self.videos_dev_path = videos_dev_path
                self.dev_data = []
                self.dev_batch_index = 0
        else:
            raise ValueError(
                'The method_flag to generate_video_paths must be an integer between 1 or 2. Value given: '
                +str(method_flag))

    def complete_batches(self):
        self.train_indexes = list(range(0, len(self.train_data)))
        self.train_batches = int( len(self.train_data) / self.batch_size)
        residuo = len(self.train_data) % self.batch_size
        if residuo != 0:
            self.train_batches += 1
            for i in range(self.batch_size - residuo):
                random_index = np.random.randint(0, len(self.train_data))
                self.train_indexes += [random_index]
                self.train_data =  np.append(self.train_data,
                                         self.train_data[random_index])

        self.test_indexes = list(range(0, len(self.test_data)))
        self.test_batches = int( len(self.test_data) /  self.batch_size)
        residuo = len(self.test_data) % self.batch_size
        if residuo != 0:
            self.test_batches += 1
            for i in range(self.batch_size - residuo):
                random_index = np.random.randint(0, len(self.test_data))
                self.test_indexes += [random_index]
                self.test_data = np.append(self.test_data,
                                        self.test_data[random_index])

        if self.dev_path:
            self.dev_indexes = list(range(0, len(self.dev_data)))
            self.dev_batches = int( len(self.dev_data) / self.batch_size)
            residuo = len(self.dev_data) % self.batch_size
            if residuo != 0:
                self.dev_batches += 1
                for i in range(self.batch_size - residuo):
                    random_index = np.random.randint(0, len(self.dev_data))
                    self.dev_indexes += [random_index]
                    self.dev_data = np.append(self.dev_data,
                                            self.dev_data[random_index])

    def shuffle_videos(self):
        self.train_indexes = np.random.permutation(self.train_indexes)
        self.test_indexes = np.random.permutation(self.test_indexes)

        if self.dev_path:
            self.dev_indexes = np.random.permutation(self.dev_indexes)

    def load_video(self, video_dictionary, channels = 3):
        video = []
        frames_path = tuple(video_dictionary.values())[0][0]
        function = tuple(video_dictionary.keys())[0][1]

        for frame in frames_path:
            image = self.load_raw_frame(frame, channels)
            image = function(image)
            video.append(image)

        video = np.asarray(video, dtype=np.float32)
        if channels == 1:
            video = video.reshape((self.video_frames,self.frame_size[1], self.frame_size[0],1))
        return video

    def get_next_train_batch(self, n_channels = 3):

        if self.train_batch_index >= self.train_batches:
            if self.shuffle:
                self.train_indexes = np.random.permutation(self.train_indexes)
            self.train_batch_index = 0

        start_index = self.train_batch_index*self.batch_size % len(self.train_indexes)
        end_index = start_index + self.batch_size

        batch = []
        labels = []
        for index in range(start_index,end_index):
            label = tuple(self.train_data[self.train_indexes[index]].values())[0][1]
            video = self.load_video(self.train_data[self.train_indexes[index]], channels=n_channels)
            labels.append(label)
            batch.append(video)

        if self.video_transformation:
            for index, callback in enumerate(self.video_transformation):
                if self.train_batch_index >= self.trans_train_indexes[index]:
                    for i in range(len(batch)):
                        batch[i] = callback(batch[i])
                        if batch[i].shape != (self.video_frames, self.frame_size[1], self.frame_size[0], n_channels):
                            raise AssertionError(
                                'The python callback given in video_transformation returns an invalid '
                                'video shape. Video dimension return: '+str(batch[i].shape)+' \nDimension '
                                'that must be return: ' + str((self.video_frames, self.frame_size[1], self.frame_size[0], n_channels))
                            )

        self.train_batch_index += 1

        return np.asarray(batch, dtype=np.float32), np.asarray(labels, dtype=np.int32)

    def get_next_test_batch(self, n_channels=3):
        if self.test_batch_index >= self.test_batches:
            if self.shuffle:
                self.test_indexes = np.random.permutation(self.test_indexes)
            self.test_batch_index = 0

        start_index = self.test_batch_index * self.batch_size % len(self.test_indexes)
        end_index = start_index + self.batch_size

        batch = []
        labels = []
        for index in range(start_index, end_index):
            label = tuple(self.test_data[self.test_indexes[index]].values())[0][1]
            video = self.load_video(self.test_data[self.test_indexes[index]], channels=n_channels)
            labels.append(label)
            batch.append(video)

        if self.video_transformation:
            for index, callback in enumerate(self.video_transformation):
                if self.test_batch_index >= self.trans_test_indexes[index]:
                    for i in range(len(batch)):
                        batch[i] = callback(batch[i])
                        if batch[i].shape != (self.video_frames, self.frame_size[1], self.frame_size[0], n_channels):
                            raise AssertionError(
                                'The python callback given in video_transformation returns an invalid '
                                'video shape. Video dimension return: '+str(batch[i].shape)+' \nDimension '
                                'that must be return: ' + str((self.video_frames, self.frame_size[1], self.frame_size[0], n_channels))
                            )

        self.test_batch_index += 1

        return np.asarray(batch, dtype=np.float32), np.asarray(labels, dtype=np.int32)

    def get_next_dev_batch(self, n_channels=3):
        if self.dev_path:
            if self.dev_batch_index >= self.dev_batches:
                if self.shuffle:
                    self.dev_indexes = np.random.permutation(self.dev_indexes)
                self.dev_batch_index = 0

            start_index = self.dev_batch_index * self.batch_size % len(self.dev_indexes)
            end_index = start_index + self.batch_size

            batch = []
            labels = []
            for index in range(start_index, end_index):
                label = tuple(self.dev_data[self.dev_indexes[index]].values())[0][1]
                video = self.load_video(self.dev_data[self.dev_indexes[index]], channels=n_channels)
                labels.append(label)
                batch.append(video)

            if self.video_transformation:
                for index, callback in enumerate(self.video_transformation):
                    if self.dev_batch_index >= self.trans_dev_indexes[index]:
                        for i in range(len(batch)):
                            batch[i] = callback(batch[i])
                            if batch[i].shape != (self.video_frames, self.frame_size[1], self.frame_size[0], n_channels):
                                raise AssertionError(
                                    'The python callback given in video_transformation returns an invalid '
                                    'video shape. Video dimension return: '+str(batch[i].shape)+' \nDimension '
                                    'that must be return: ' + str((self.video_frames, self.frame_size[1], self.frame_size[0], n_channels))
                                )

            self.dev_batch_index += 1

            return np.asarray(batch, dtype=np.float32), np.asarray(labels, dtype=np.int32)
        else:
            raise AttributeError(
                'No se puede llamar a la funcion debido a que en el directorio no se'
                'encuentra la carpeta dev y por ende no se tienen datos en dev'
            )

    def get_train_generator(self, channels = 3):
        while True:
            yield self.get_next_train_batch(channels)

    def get_test_generator(self, channels = 3):
        while True:
            yield self.get_next_test_batch(channels)

    def get_dev_generator(self, channels = 3):
        while True:
            yield self.get_next_dev_batch(channels)

    def load_raw_frame(self,frame_path, channels = 3, original_size_created = True):
        if channels == 1:
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        elif channels == 3:
            img =  cv2.cvtColor(cv2.imread(frame_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if original_size_created:
            return cv2.resize(img, tuple(self.original_size))
        else:
            return img

    def resize_frame(self, image):
        return cv2.resize(image, tuple(self.frame_size))

    def sequential_temporal_crop(self, video_path, video_index, list_name, method_flag):
        if not isinstance(video_path, str):
            raise ValueError(
                'Video_path is different from a string. Instance given: {i}'.format(
                    i=type(video_path)
                    )
                )
        video = video_path
        frames_path = [os.path.join(video, frame) for frame in sorted(os.listdir(video))]
        while len(frames_path) < self.video_frames:
            frames_path += frames_path[:self.video_frames - len(frames_path)]
        if method_flag == 1:
            label = self.to_number[video.split("/")[-2].lower()]
        elif method_flag == 2:
            if list_name == "train":
                label = self.to_number[str(self.df[self.train_indexes[video_index],2]).lower()]
            elif list_name == "test":
                label = self.to_number[str(self.df[self.test_indexes[video_index], 2]).lower()]
            elif list_name == "dev":
                label = self.to_number[str(self.df[self.dev_indexes[video_index], 2]).lower()]
            else:
                raise ValueError(
                'The list name given is invalid. List name given {i}'.format(i=list_name)
                )
        else:
            raise ValueError(
                'The method_flag to sequential_temporal_crop must be an integer between 1 or 2. Value given: '
                +str(method_flag))

        n_veces = len(frames_path) // self.video_frames
        for i in range(n_veces):
            start = self.video_frames * i
            end = self.video_frames * (i + 1)
            frames = frames_path[start:end]

            name = "tcrop" + str(self.transformation_index)
            elemento = {(name, None): (frames, label)}
            self.transformation_index += 1
            if list_name == 'train':
                self.train_data.append(elemento)
            elif list_name == 'test':
                self.test_data.append(elemento)
            elif list_name == 'dev':
                self.dev_data.append(elemento)
            else:
                raise ValueError(
                'The list name given is invalid. List name given {i}'.format(i=list_name)
                )

    def random_temporal_crop(self, video_path, video_index, list_name, n_veces, method_flag):
        if not isinstance(video_path, str):
            raise ValueError(
                'Video_path is different from a string. Instance given: {i}'.format(
                    i=type(video_path)
                    )
                )
        video = video_path
        frames_path = [os.path.join(video, frame) for frame in sorted(os.listdir(video))]
        while len(frames_path) < self.video_frames:
            frames_path += frames_path[:self.video_frames - len(frames_path)]
        if method_flag == 1:
            label = self.to_number[video.split("/")[-2].lower()]
        elif method_flag == 2:
            if list_name == "train":
                label = self.to_number[str(self.df[self.train_indexes[video_index], 2]).lower()]
            elif list_name == "test":
                label = self.to_number[str(self.df[self.test_indexes[video_index], 2]).lower()]
            elif list_name == "dev":
                label = self.to_number[str(self.df[self.dev_indexes[video_index], 2]).lower()]
            else:
                raise ValueError(
                'The list name given is invalid. List name given {i}'.format(i=list_name)
                )
        else:
            raise ValueError(
                'The method_flag to random_temporal_crop must be an integer between 1 or 2. Value given: '
                +str(method_flag))

        for _ in range(n_veces):
            start = np.random.randint(0, len(frames_path)-self.video_frames)
            end = start + self.video_frames
            frames = frames_path[start: end]

            name = "tcrop" + str(self.transformation_index)
            elemento = {(name, None): (frames, label)}
            self.transformation_index += 1
            if list_name == 'train':
                self.train_data.append(elemento)
            elif list_name == 'test':
                self.test_data.append(elemento)
            elif list_name == 'dev':
                self.dev_data.append(elemento)
            else:
                raise ValueError(
                'The list name given is invalid. List name given {i}'.format(i=list_name)
                )

    def custom_temporal_crop(self, video_path, video_index, list_name, custom_fn, method_flag):
        if not isinstance(video_path, str):
            raise ValueError(
                'Video_path is different from a string. Instance given: {i}'.format(
                    i=type(video_path)
                    )
                )
        video = video_path
        frames_path = [os.path.join(video, frame) for frame in sorted(os.listdir(video))]
        while len(frames_path) < self.video_frames:
            frames_path += frames_path[:self.video_frames - len(frames_path)]
        try:
            frames = custom_fn(frames_path)
            if method_flag == 1:
                label = self.to_number[video.split("/")[-2].lower()]
            elif method_flag == 2:
                if list_name == "train":
                    label = self.to_number[str(self.df[self.train_indexes[video_index], 2]).lower()]
                elif list_name == "test":
                    label = self.to_number[str(self.df[self.test_indexes[video_index], 2]).lower()]
                elif list_name == "dev":
                    label = self.to_number[str(self.df[self.dev_indexes[video_index], 2]).lower()]
                else:
                    raise ValueError(
                'The list name given is invalid. List name given {i}'.format(i=list_name)
                )
            else:
                raise ValueError(
                'The method_flag to custom_temporal_crop must be an integer between 1 or 2. Value given: '
                +str(method_flag))
            n_veces = len(frames)
            for i in range(n_veces):
                if len(frames[i]) != self.video_frames:
                    raise ValueError(
                        'The number of frames to add with custom function must be '
                        'equal to specified by user. Found len: %s' % len(frames[i])
                    )
                name = "tcrop" + str(self.transformation_index)
                elemento = {(name, None): (frames[i], label)}
                self.transformation_index += 1
                if list_name == 'train':
                    self.train_data.append(elemento)
                elif list_name == 'test':
                    self.test_data.append(elemento)
                elif list_name == 'dev':
                    self.dev_data.append(elemento)
                else:
                    raise ValueError(
                'The list name given is invalid. List name given {i}'.format(i=list_name)
                )

        except:
            raise AttributeError(
                'Custom_temporal_crop should return a python list of list '
                'where every row if a temporal crop on a video and the length '
                'of every row is the video frames length specified by user.'
            )

    def none_temporal_crop(self, video_path, video_index, list_name, method_flag):
        if not isinstance(video_path, str):
            raise ValueError(
                'Video_path is different from a string. Instance given: {i}'.format(
                    i=type(video_path)
                    )
                )
        video = video_path
        name = "tcrop" + str(self.transformation_index)
        frames_path = [os.path.join(video, frame) for frame in sorted(os.listdir(video))]
        while len(frames_path) < self.video_frames:
            frames_path += frames_path[:self.video_frames - len(frames_path)]
        frames_path = frames_path[:self.video_frames]
        if method_flag == 1:
            label = self.to_number[video.split("/")[-2].lower()]
        elif method_flag == 2:
            if list_name == "train":
                label = self.to_number[str(self.df[self.train_indexes[video_index], 2]).lower()]
            elif list_name == "test":
                label = self.to_number[str(self.df[self.test_indexes[video_index], 2]).lower()]
            elif list_name == "dev":
                label = self.to_number[str(self.df[self.dev_indexes[video_index], 2]).lower()]
            else:
                raise ValueError(
                'The list name given is invalid. List name given {i}'.format(i=list_name)
                )
        else:
            raise ValueError(
                'The method_flag to none_temporal_crop must be an integer between 1 or 2. Value given: '
                +str(method_flag))
        elemento = {(name, None): (frames_path, label)}
        self.transformation_index += 1
        if list_name == 'train':
            self.train_data.append(elemento)
        elif list_name == 'test':
            self.test_data.append(elemento)
        elif list_name == 'dev':
            self.dev_data.append(elemento)
        else:
            raise ValueError(
                'The list name given is invalid. List name given {i}'.format(i=list_name)
                )

    def temporal_crop(self, mode , custom_fn, method_flag):
        if mode == 'sequential':
            for index, video in enumerate(self.videos_train_path):
                self.sequential_temporal_crop(video, index, "train", method_flag)

            for index, video in enumerate(self.videos_test_path):
                self.sequential_temporal_crop(video,index, "test", method_flag)

            if self.dev_path:
                for index, video in enumerate(self.videos_dev_path):
                    self.sequential_temporal_crop(video,index,"dev",method_flag)

        elif mode == 'random':
            if isinstance(custom_fn, int):
                n_veces = custom_fn
            else:
                raise ValueError(
                    'When you use the random temporal crop, the parameter must be an Integer. '
                    'Value given: %s' % type(custom_fn)
                )
            for index, video in enumerate(self.videos_train_path):
                self.random_temporal_crop(video, index, "train", n_veces, method_flag)

            for index, video in enumerate(self.videos_test_path):
                self.random_temporal_crop(video, index, "test", n_veces, method_flag)

            if self.dev_path:
                for index, video in enumerate(self.videos_dev_path):
                    self.random_temporal_crop(video, index, "dev", n_veces, method_flag)

        elif mode == 'custom':
            if custom_fn:
                for index, video in enumerate(self.videos_train_path):
                    self.custom_temporal_crop(video,index, "train",custom_fn, method_flag)

                for index, video in enumerate(self.videos_test_path):
                    self.custom_temporal_crop(video, index, "test", custom_fn, method_flag)

                if self.dev_path:
                    for index, video in enumerate(self.videos_dev_path):
                        self.custom_temporal_crop(video, index, "dev", custom_fn, method_flag)
            else:
                raise ValueError(
                    'When you use custom temporal crop, the parameter must be a python '
                    'callback. Parameter given: %s' % type(custom_fn)
                    )

        else:
            for index, video in enumerate(self.videos_train_path):
                self.none_temporal_crop(video, index, "train", method_flag)

            for index, video in enumerate(self.videos_test_path):
                self.none_temporal_crop(video, index, "test", method_flag)

            if self.dev_path:
                for index, video in enumerate(self.videos_dev_path):
                    self.none_temporal_crop(video, index, "dev", method_flag)

    def sequential_frame_crop(self,list_name, conserve):
        if list_name == 'train':
            lista = self.train_data
        elif list_name == 'test':
            lista = self.test_data
        elif list_name == 'dev':
            lista = self.dev_data
        else:
            raise ValueError(
                'The list name given is invalid. List name given {i}'.format(i=list_name)
                )
        original_height = self.original_size[1]
        original_width = self.original_size[0]
        new_lista = []

        for video in lista:
            # Agrego los nuevos cortes de frames a los datos
            values = tuple(video.values())[0]
            n_veces = [original_width // self.frame_size[0], original_height // self.frame_size[1]]

            for i in range(n_veces[0]):
                for j in range(n_veces[1]):
                    start_width = i * self.frame_size[0]
                    end_width = start_width + self.frame_size[0]
                    start_height = j * self.frame_size[1]
                    end_height = start_height + self.frame_size[1]
                    function = lambda frame: frame[start_height: end_height, start_width: end_width]

                    name = "icrop" + str(self.transformation_index)
                    self.transformation_index += 1
                    elemento = {(name, function): values}
                    new_lista.append(elemento)

        if conserve:
            self.none_frame_crop(list_name)
            if list_name == 'train':
                self.train_data += new_lista
            elif list_name == 'test':
                self.test_data += new_lista
            elif list_name == 'dev':
                self.dev_data += new_lista
        else:
            if list_name == 'train':
                self.train_data = new_lista
            elif list_name == 'test':
                self.test_data = new_lista
            elif list_name == 'dev':
                self.dev_data = new_lista

    def random_frame_crop(self,list_name, conserve, n_veces):
        if list_name == 'train':
            lista = self.train_data
        elif list_name == 'test':
            lista = self.test_data
        elif list_name == 'dev':
            lista = self.dev_data
        else:
            raise ValueError(
                'The list name given is invalid. List name given {i}'.format(i=list_name)
                )
        original_height = self.original_size[1]
        original_width = self.original_size[0]
        new_lista = []

        for video in lista:
            # Agrego los nuevos cortes de frames a los datos
            values = tuple(video.values())[0]

            for _ in range(n_veces):
                start_width = np.random.randint(0, original_width - self.frame_size[0])
                end_width = start_width + self.frame_size[0]
                start_height = np.random.randint(0, original_height - self.frame_size[1])
                end_height = start_height + self.frame_size[1]
                function = lambda frame: frame[start_height: end_height, start_width: end_width]

                name = "icrop" + str(self.transformation_index)
                self.transformation_index += 1
                elemento = {(name, function): values}
                new_lista.append(elemento)

        if conserve:
            self.none_frame_crop(list_name)
            if list_name == 'train':
                self.train_data += new_lista
            elif list_name == 'test':
                self.test_data += new_lista
            elif list_name == 'dev':
                self.dev_data += new_lista
        else:
            if list_name == 'train':
                self.train_data = new_lista
            elif list_name == 'test':
                self.test_data = new_lista
            elif list_name == 'dev':
                self.dev_data = new_lista

    def custom_frame_crop(self,list_name, conserve, custom_fn):
        if list_name == 'train':
            lista = self.train_data
        elif list_name == 'test':
            lista = self.test_data
        elif list_name == 'dev':
            lista = self.dev_data
        else:
            raise ValueError(
                'The list name given is invalid. List name given {i}'.format(i=list_name)
                )
        original_height = self.original_size[1]
        original_width = self.original_size[0]
        new_lista = []

        for video in lista:
            # Agrego los nuevos cortes de frames a los datos
            values = tuple(video.values())[0]

            try:
                cortes = custom_fn(original_width, original_height)
                for corte in cortes:
                    size_frame = (corte[1] - corte[0], corte[3] - corte[2])
                    if size_frame[0] != self.frame_size[0] or size_frame[1] != self.frame_size[1]:
                        raise ValueError(
                            'The final frame size must be equal at the size specified '
                            'by user. Size returned %s' % str(size_frame)
                        )
                    function = lambda frame: frame[corte[2]: corte[3], corte[0]: corte[1]]
                    name = "icrop" + str(self.transformation_index)
                    self.transformation_index += 1
                    elemento = {(name, function): values}
                    new_lista.append(elemento)

            except:
                raise AttributeError(
                    'Custom_frame_crop should return a python list of list '
                    'where every row if a frame crop on frames and the length '
                    'of every row must be 4 (start_width_point, end_with_point, '
                    'start_height_point, end_height_point) in that order.'
                )
        if conserve:
            self.none_frame_crop(list_name)
            if list_name == 'train':
                self.train_data += new_lista
            elif list_name == 'test':
                self.test_data += new_lista
            elif list_name == 'dev':
                self.dev_data += new_lista
        else:
            if list_name == 'train':
                self.train_data = new_lista
            elif list_name == 'test':
                self.test_data = new_lista
            elif list_name == 'dev':
                self.dev_data = new_lista

    def none_frame_crop(self, list_name):
        """Metodo que se encarga de realizar el corte espacial None en
        un video dado y se modificara en la lista indicada.
        Args:
            list_name: String con las opciones ("train", "test", "dev") para escoger
                              a que lista se agregaran los videos de formas None.
        """
        if list_name == 'train':
            lista = self.train_data
        elif list_name == 'test':
            lista = self.test_data
        elif list_name == 'dev':
            lista = self.dev_data
        else:
            raise ValueError(
                'The list name given is invalid. List name given {i}'.format(i=list_name)
                )
        for index in range(len(lista)):
            llave_original = tuple(lista[index].keys())[0]
            llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
            valores = tuple(lista[index].values())[0]
            lista[index] = {llave_nueva: valores}

    def frame_crop(self,mode, custom_fn, conserve_original = False):
        if mode == 'sequential':
            self.sequential_frame_crop("train", conserve_original)
            self.sequential_frame_crop("test", conserve_original)
            if self.dev_path:
                self.sequential_frame_crop("dev", conserve_original)

        elif mode == 'random':
            if isinstance(custom_fn, int):
                n_veces = custom_fn
            else:
                raise ValueError(
                    'When you use the random frame crop, the parameter must be an Integer. '
                    'Value given: %s' % type(custom_fn)
                )
            self.random_frame_crop("train",conserve_original, n_veces)
            self.random_frame_crop("test", conserve_original, n_veces)
            if self.dev_path:
                self.random_frame_crop("dev", conserve_original, n_veces)

        elif mode == 'custom':
            if custom_fn:
                self.custom_frame_crop("train", conserve_original, custom_fn)
                self.custom_frame_crop("test", conserve_original, custom_fn)
                if self.dev_path:
                    self.custom_frame_crop("dev", conserve_original, custom_fn)
            else:
                raise ValueError(
                    'When you use custom frame crop, the parameter must be a python '
                    'callback. Parameter given: %s' % type(custom_fn)
                    )

        else:
            self.none_frame_crop("train")
            self.none_frame_crop("test")
            if self.dev_path:
                self.none_frame_crop("dev")
