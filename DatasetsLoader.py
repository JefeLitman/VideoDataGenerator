"""Video Data Generator for Any Video Dataset with Custom Transformations
You can use it in your own and only have three dependencies with opencv, numpy and pandas.
Version: 2.2.3
"""

import os
import cv2
import numpy as np
import pandas as pd

class VideoDataGenerator():
    """Clase para cargar todos los datos de un Dataset a partir de la ruta
    especificada por el usuario y ademas, agregando transformaciones en
    tiempo real.

    En esta version lee los datos a partir de los frames pero no archivos avi.
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
                 conserve_original = False
                 ):
        """Constructor de la clase.
        Args:
            directory_path: String que tiene la ruta absoluta de la ubicacion del
                                       dataset, incluyendo en el path el split. Obligatorio si no se
                                       usa el parametro de table_paths.
            table_paths: Pandas Dataframe, numpy array o python matrix con forma (nro_videos, 3) donde las
                                          columnas son (path_video, tipo_video, clase o etiqueta) y tipo de video puede ser
                                          "train", "test" o "dev" unicamente. Obligatorio si no se usa el parametro directory_path
            batch_size: Numero que corresponde al tamaño de elementos por batch. Cuando sea None va tomar
                                como batch, el tamaño de la menor cantidad de datos del dataset o si la menor cantidad
                                es mayor que 32 se pondra si esta en None. Por defecto en None.
            original_frame_size: Tupla de enteros del tamaño de imagen original a cargar con la estructura (width, height).
                                                Por defecto en None y tomara el tamaño original de las imagenes.
            frame_size: Tupla de enteros del tamaña final de imagen que quedara con la estructura (width, height).
                                 Por defecto en None y tomara el tamaño original de las imagenes.
            video_frames: Numero de frames con los que quedara los batch.
                                    Por defecto 16.
            temporal_crop: Tupla de la forma (Modo, funcion customizada o callback)
                                    de python e indica que tipo de corte temporal se hara sobre los videos.
                                    Por defecto en (None, None).
            video_transformation: Lista de python que contiene tuplas de la forma (aplicabilidad, callback python).
                                                 La aplicabilidad puede ser "full" o "augmented". El orden que lleve la lista es como
                                                 se aplican las transformaciones. Por defecto en None
            frame_crop: Tupla de la forma (Modo, funcion customizada o callback)
                                 de python e indica el tipo de corte para cada frame de cada video
                                que se hara sobre los videos. Por defecto en (None, None).
            shuffle: Booleano que determina si deben ser mezclados aleatoreamente los datos.
                        Por defecto en False.
            conserve_original: Booleano que determina si se debe guardar los datos originales
                                          junto a los datos transformados para entregarlos. Por defecto
                                          en False

            Aclaratoria es que esta clase incluye por defecto las carpetas de train, test y dev,
            ademas, siempre usa la notacion de canales al final"""

        """Definicion de constantes, atributos y restricciones a los parametros"""
        temporal_crop_modes = (None,'sequential','random','custom')
        frame_crop_modes = (None,'sequential','random','custom')

        method_flag = 0
        self.transformation_index = 0
        self.dev_path = None
        self.dev_data = None

        """Proceso de revisar que los directorios del path esten en la jerarquia correcta si se usa un directorio"""
        if directory_path is not None:
            method_flag = 1
            self.using_folders(directory_path)

        """Proceso de revisar que el table_paths este bien formado"""
        if table_paths is not None:
            if method_flag == 1:
                raise ValueError('Debe usar unicamente uno de los dos metodos para cargar los datos pero '
                                 'no ambos. Valores pasados: {dp} junto a {df}'.format(dp=directory_path, df=table_paths))
            method_flag = 2
            self.using_table_paths(table_paths)

        if method_flag == 0:
            raise ValueError('Debe usar al menos un metodo para cargar los datos, o bien por directorio de carpetas '
                             'o por medio de una tabla con las rutas de las carpetas. Ambos valores pasados fueron None.')

        """Proceso de revisar que las transformaciones escojidas son validas"""
        if temporal_crop[0] not in temporal_crop_modes:
            raise ValueError(
                'Los unicos modos disponibles a usar para corte temporal son '
                '(None, sequential, random, custom). El modo escojido no es valido '
                'por que tomo la opcion %s' % temporal_crop[0]
            )
        if frame_crop[0] not in frame_crop_modes:
            raise ValueError(
                'Los unicos modos disponibles a usar son para corte de imagenes son '
                '(None, sequential, random, custom). El modo escojido no es valido '
                'por que tomo la opcion %s' % frame_crop[0]
            )

        """Proceso de generar los path de los videos para la verificacion subsiguiente de parametros"""
        self.generate_classes(method_flag)
        self.generate_videos_paths(method_flag)

        """Proceso de definir el tamaño de batch si fue especificado como None"""
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

        """Proceso de definir el tamaño original de todas las imagenes si no se entrega
        el parametro de original size y establecer el tamaño de los frames"""
        if original_frame_size:
            self.original_size = original_frame_size
        else:
            frames_path = os.path.join(self.videos_train_path[0], sorted(os.listdir(self.videos_train_path[0]))[0])
            self.original_size = self.load_raw_frame(frames_path, original_size_created=False).shape[1::-1]
        if frame_size:
            self.frame_size = frame_size
        else:
            self.frame_size = self.original_size

        """Proceso de revisar que los tamaños de frame quedaron bien especificados"""
        if frame_crop[0] == 'sequential' and conserve_original == False:
            if min([self.original_size[0] // self.frame_size[0], self.original_size[1] // self.frame_size[1]]) == 0:
                raise ValueError('No es posible realizar la transformacion secuencial sobre los '
                                 'frames ya que el tamaño de salida ({w}, {h}) es mayor al original y '
                                 'los datos originales no seran salvados por lo que tendra todos los datos vacios.'.format(w = self.frame_size[0],
                                                                                                               h = self.frame_size[1]))

        """Proceso de establecer el valor de video frames si se establecio en None con los minimos frames de todos los videos"""
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

        """Proceso de generar los datos con o sin transformaciones"""
        if conserve_original and temporal_crop[0] not in (None, 'sequential'):
            self.temporal_crop(mode = 'sequential', custom_fn=temporal_crop[1], method_flag=method_flag)
        self.temporal_crop(mode = temporal_crop[0], custom_fn=temporal_crop[1], method_flag=method_flag)
        self.frame_crop(mode=frame_crop[0], custom_fn=frame_crop[1], conserve_original=conserve_original)

        """Proceso de realizar el desordenamiento de los datos"""
        if shuffle:
            self.shuffle_videos()

        """Proceso de verificar que las transformaciones sobre los videos esten bien"""
        if video_transformation:
            self.train_indexes = []
            self.test_indexes = []
            if self.dev_path:
                self.dev_indexes = []
            try:
                for mode, _ in video_transformation:
                    if mode.lower() == "augmented":
                        self.train_indexes.append(len(self.train_data) // self.batch_size)
                        self.train_data = np.append(self.train_data, self.train_data)
                        self.test_indexes.append(len(self.test_data) // self.batch_size)
                        self.test_data = np.append(self.test_data, self.test_data)
                        if self.dev_path:
                            self.dev_indexes.append(len(self.dev_data) // self.batch_size)
                            self.dev_data = np.append(self.dev_data, self.dev_data)
                    elif mode.lower() == "full":
                        self.train_indexes.append(0)
                        self.test_indexes.append(0)
                        if self.dev_path:
                            self.dev_indexes.append(0)
                    else:
                        raise AttributeError('Se ha especificado una transformacion con modo no valido. Los valores '
                                             'validos son "full" o "augmented". Valor pasado: '+str(mode))
            except:
                raise ValueError('El parametro de video_transformation debe ser una lista de python con tuplas que contienen 2 valores. Estas tuplas '
                                 'tienen la forma (modo_aplicacion, callback python) donde modo puede ser "full" o "augmented". Parametro pasado: '+str(video_transformation))
            self.video_transformation = [pair[1] for pair in video_transformation]
        else:
            self.video_transformation = video_transformation

        """Por ultimo el proceso de completar los batches en caso de hacer falta"""
        self.complete_batches()

    def using_folders(self, directory_path):
        """Metodo que se encarga de construir los path de los videos a partir del
        directorio pasado por el usuario.
         Args:
             directory_path: String que tiene la ruta absoluta de la ubicacion del
                                       dataset, incluyendo en el path el split.
             """
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
                    'La organizacion de la carpeta debe seguir la estructura '
                    'de train, test y dev (este ultimo opcional) teniendo '
                    'los mismos nombres sin importar mayus o minus. '
                    'Carpeta del error: %s' % i)

    def using_table_paths(self, table_paths):
        """Metodo que se encarga de construir los paths de los videos a partir de una matrix
        o tabla que contiene los paths, tipo de video y clase del mismo en ella.
        Args:
            table_paths: Dataframe, numpy array o python matrix con forma (nro_videos, 3) donde las
                                  columnas son (path_video, tipo_video, clase o etiqueta) y tipo de video puede ser
                                  "train", "test" o "dev" unicamente.
            """
        if isinstance(table_paths, list):
            self.df = np.r_[table_paths]
        elif isinstance(table_paths, pd.DataFrame):
            self.df = table_paths.values
        elif isinstance(table_paths, np.ndarray):
            self.df = table_paths
        else:
            raise ValueError(
                'El parametro de table_paths debe ser de tipo lista de python, array de numpy o DataFrame de '
                'pandas unicamente. Tipo de table_paths pasado: ' + str(type(table_paths)))

        if self.df.shape[1] != 3 and self.df.ndim != 2:
            raise ValueError(
                'table_paths o no es una matriz o no tiene 3 columnas. Dimensiones pasadas: ' + str(self.df.shape))

    def generate_classes(self, method_flag):
        """Metodo que se encarga de generar el numero de clases, los nombres
        de clases, numeros con indices de clase y el diccionario que convierte de clase
        a numero como de numero a clase.
        Args:
            method_flag: Variable numerica que me indica en que forma estoy cargando los datos.
        """
        if method_flag == 1:
            self.to_class = [clase.lower() for clase in sorted(os.listdir(self.train_path))] #Equivale al vector de clases
        elif method_flag == 2:
            self.to_class = [str(clase).lower() for clase in np.unique(self.df[:,2])]
        else:
            raise ValueError('El valor pasado a la funcion generate_classes con el parametro method_flag debe ser un numero '
                             'entre 1 y 2. Valor pasado: '+str(method_flag))
        self.to_number = dict((name, index) for index, name in enumerate(self.to_class))

    def generate_videos_paths(self, method_flag):
        """Metodo que se encarga de generar una lista con el path absoluto de todos los videos para train, test y
        dev si llega a aplicar esta carpeta."""

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
                        'Dentro de la estructura de table_paths existe un tipo de video no compatible. '
                        'Los valores permitidos son "train", "test" y "dev". Valor del error: ' + str(video_param[1]))

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
                'El valor pasado a la funcion generate_video_paths con el parametro method_flag debe ser un numero '
                'entre 1 y 2. Valor pasado: ' + str(method_flag))

    def complete_batches(self):
        self.train_batches = int( len(self.train_data) / self.batch_size)
        residuo = len(self.train_data) % self.batch_size
        if residuo != 0:
            self.train_batches += 1
            random_index = np.random.randint(0, len(self.train_data) - self.batch_size)
            self.train_data =  np.append(self.train_data,
                                         self.train_data[random_index:random_index + self.batch_size - residuo])

        self.test_batches = int( len(self.test_data) /  self.batch_size)
        residuo = len(self.test_data) % self.batch_size
        if residuo != 0:
            self.test_batches += 1
            random_index = np.random.randint(0, len(self.test_data) - self.batch_size)
            self.test_data = np.append(self.test_data,
                                        self.test_data[random_index:random_index + self.batch_size - residuo])

        if self.dev_path:
            self.dev_batches = int( len(self.dev_data) / self.batch_size)
            residuo = len(self.dev_data) % self.batch_size
            if residuo != 0:
                self.dev_batches += 1
                random_index = np.random.randint(0, len(self.dev_data) - self.batch_size)
                self.dev_data = np.append(self.dev_data,
                                            self.dev_data[random_index:random_index + self.batch_size - residuo])

    def shuffle_videos(self):
        """Metodo que se encarga de realizar shuffle a los datos si esta
        activada la opcion de shuffle."""
        self.train_data = np.random.permutation(self.train_data)
        self.test_data = np.random.permutation(self.test_data)

        if self.dev_path:
            self.dev_data = np.random.permutation(self.dev_data)

    def load_video(self, video_dictionary, channels = 3):
        """Metodo que se encarga de cargar en memoria los frames de un video a partir del
        diccionario que contiene todos los elementos del video
        Args:
            video_dictionary: Diccionario que contiene el video de la forma { (nombre_operacion,
            funcion_frames) : ([frames_paths],etiqueta) }.
            canales: Numero de canales que tienen las imagenes en las carpetas. Debe ser uniforme.
            """
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

    def get_next_train_batch(self, n_canales = 3):
        """Metodo que se encarga de retornar el siguiente batch o primer batch
        de datos train si se cumple un epoch.
        Args:
            n_canales: Numero que corresponde al numero de canales que posee las imagenes. Por defecto en 3.
            """

        if self.train_batch_index >= self.train_batches:
            self.train_batch_index = 0

        start_index = self.train_batch_index*self.batch_size
        end_index = (self.train_batch_index + 1)*self.batch_size

        batch = []
        labels = []
        for index in range(start_index,end_index):
            label = tuple(self.train_data[index].values())[0][1]
            video = self.load_video(self.train_data[index], channels=n_canales)
            labels.append(label)
            batch.append(video)

        if self.video_transformation:
            for index, callback in enumerate(self.video_transformation):
                if self.train_batch_index >= self.train_indexes[index]:
                    for i in range(len(batch)):
                        batch[i] = callback(batch[i])
                        if batch[i].shape != (self.video_frames, self.frame_size[1], self.frame_size[0], n_canales):
                            raise AssertionError('La funcion pasada a video_transformation no retorna las dimensiones '
                                                 'esperadas que deberia tener el video. Dimension de video: '+str(batch[i].shape)+
                                                 ' Dimensiones que deberia tener: ' + str((self.video_frames, self.frame_size[1], self.frame_size[0], n_canales))
                            )

        self.train_batch_index += 1

        return np.asarray(batch, dtype=np.float32), np.asarray(labels, dtype=np.int32)

    def get_next_test_batch(self, n_canales=3):
        """Metodo que se encarga de retornar el siguiente batch o primer batch
        de datos test si se cumple un epoch.
        Args:
            n_canales: Numero que corresponde al numero de canales que posee las imagenes. Por defecto en 3.
            """

        if self.test_batch_index >= self.test_batches:
            self.test_batch_index = 0

        start_index = self.test_batch_index * self.batch_size
        end_index = (self.test_batch_index + 1) * self.batch_size

        batch = []
        labels = []
        for index in range(start_index, end_index):
            label = tuple(self.test_data[index].values())[0][1]
            video = self.load_video(self.test_data[index], channels=n_canales)
            labels.append(label)
            batch.append(video)

        if self.video_transformation:
            for index, callback in enumerate(self.video_transformation):
                if self.test_batch_index >= self.test_indexes[index]:
                    for i in range(len(batch)):
                        batch[i] = callback(batch[i])
                        if batch[i].shape != (self.video_frames, self.frame_size[1], self.frame_size[0], n_canales):
                            raise AssertionError('La funcion pasada a video_transformation no retorna las dimensiones '
                                                 'esperadas que deberia tener el video. Dimension de video: '+str(batch[i].shape)+
                                                 ' Dimensiones que deberia tener: ' + str((self.video_frames, self.frame_size[1], self.frame_size[0], n_canales))
                            )

        self.test_batch_index += 1

        return np.asarray(batch, dtype=np.float32), np.asarray(labels, dtype=np.int32)

    def get_next_dev_batch(self, n_canales=3):
        """Metodo que se encarga de retornar el siguiente batch o primer batch
                de datos dev si se cumple un epoch.
                Args:
                    n_canales: Numero que corresponde al numero de canales que posee las imagenes. Por defecto en 3.
                    """
        if self.dev_path:
            if self.dev_batch_index >= self.dev_batches:
                self.dev_batch_index = 0

            start_index = self.dev_batch_index * self.batch_size
            end_index = (self.dev_batch_index + 1) * self.batch_size

            batch = []
            labels = []
            for index in range(start_index, end_index):
                label = tuple(self.dev_data[index].values())[0][1]
                video = self.load_video(self.dev_data[index], channels=n_canales)
                labels.append(label)
                batch.append(video)

            if self.video_transformation:
                for index, callback in enumerate(self.video_transformation):
                    if self.dev_batch_index >= self.train_indexes[index]:
                        for i in range(len(batch)):
                            batch[i] = callback(batch[i])
                            if batch[i].shape != (self.video_frames, self.frame_size[1], self.frame_size[0], n_canales):
                                raise AssertionError(
                                    'La funcion pasada a video_transformation no retorna las dimensiones '
                                    'esperadas que deberia tener el video. Dimension de video: ' + str(batch[i].shape) +
                                    ' Dimensiones que deberia tener: ' + str(
                                        (self.video_frames, self.frame_size[1], self.frame_size[0], n_canales))
                                    )

            self.dev_batch_index += 1

            return np.asarray(batch, dtype=np.float32), np.asarray(labels, dtype=np.int32)
        else:
            raise AttributeError(
                'No se puede llamar a la funcion debido a que en el directorio no se'
                'encuentra la carpeta dev y por ende no se tienen datos en dev'
            )

    def get_train_generator(self, n_canales = 3):
        """Metodo que se encarga de retornar el generador de los batches de
        entrenamiento retornardo una tupla de batch y label.
        Args:
            n_canales: Numero que corresponde al numero de canales que posee las imagenes. Por defecto en 3.
        """
        while True:
            yield self.get_next_train_batch(n_canales)

    def get_test_generator(self, n_canales = 3):
        """Metodo que se encarga de retornar el generador de los batches de
        testeo retornardo una tupla de batch y label.
        Args:
            n_canales: Numero que corresponde al numero de canales que posee las imagenes. Por defecto en 3.
        """
        while True:
            yield self.get_next_test_batch(n_canales)

    def get_dev_generator(self, n_canales = 3):
        """Metodo que se encarga de retornar el generador de los batches de
        dev retornardo una tupla de batch y label.
        Args:
            n_canales: Numero que corresponde al numero de canales que posee las imagenes. Por defecto en 3.
        """
        while True:
            yield self.get_next_dev_batch(n_canales)

    def load_raw_frame(self,frame_path, channels = 3, original_size_created = True):
        """Metodo que se encarga de cargar los frames dada la ruta en memoria
        Args:
            frame_path: String que posee la ruta absoluta del frame
            channels: Entero opcional, que corresponde a cuantos canales desea cargar la imagen"""
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
        """Metodo que se encarga de redimensionar un frame segun el tamaño
        especificado por el usuario"""
        return cv2.resize(image, tuple(self.frame_size))

    def sequential_temporal_crop(self, video_path, video_index, list_name, method_flag):
        """Metodo que se encarga de realizar el corte temporal secuencial en
       un video dado por su path y se agregara a la lista indicada.
        Args:
            video_path: String del ath donde se encuentran los frames del video.
            video_index: Entero que corresponde al indice en el df del video, solo se usa
                                 cuando esta method_flag en 2.
            list_name: String con las opciones ("train", "test", "dev") para escoger
                              a que lista se agregaran los videos de formas secuencial.
             method_flag: Variable numerica que me indica en que forma estoy cargando los datos.
        """
        if not isinstance(video_path, str):
            raise ValueError('Se ha pasado video_path como una instancia diferente'
                             'a lo que es un string. Instancia pasada: {i}'.format(i=type(video_path)))
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
                raise ValueError('Se ha pasado un modo invalido al cual no se le '
                                 'pueden asignar el label del  elemento a los datos de train, test '
                                 'y dev. Modo pasado: {i}'.format(i=list_name))
        else:
            raise ValueError('El valor pasado a la funcion generate_classes con el parametro method_flag debe ser un numero '
                             'entre 1 y 2. Valor pasado: '+str(method_flag))

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
                raise ValueError('Se ha pasado un modo invalido al cual no se le '
                                 'pueden agregar elementos a los datos de train, test '
                                 'y dev. Modo pasado: {i}'.format(i=list_name))

    def random_temporal_crop(self, video_path, video_index, list_name, n_veces, method_flag):
        """Metodo que se encarga de realizar el corte temporal aleatorio en
        un video dado por su path y se agregara a la lista indicada.
                Args:
                    video_path: String del ath donde se encuentran los frames del video.
                    video_index: Entero que corresponde al indice en el df del video, solo se usa
                                         cuando esta method_flag en 2.
                    list_name: String con las opciones ("train", "test", "dev") para escoger
                                      a que lista se agregaran los videos de formas aleatoria.
                    n_veces: Entero que indica cuantos cortes temporales aleatorios se
                                   deben hace sobre el video.
                    method_flag: Variable numerica que me indica en que forma estoy cargando los datos.
        """
        if not isinstance(video_path, str):
            raise ValueError('Se ha pasado video_path como una instancia diferente'
                             'a lo que es un string. Instancia pasada: {i}'.format(i=type(video_path)))
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
                raise ValueError('Se ha pasado un modo invalido al cual no se le '
                                 'pueden asignar el label del  elemento a los datos de train, test '
                                 'y dev. Modo pasado: {i}'.format(i=list_name))
        else:
            raise ValueError(
                'El valor pasado a la funcion generate_classes con el parametro method_flag debe ser un numero '
                'entre 1 y 2. Valor pasado: ' + str(method_flag))

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
                raise ValueError('Se ha pasado un modo invalido al cual no se le '
                                 'pueden agregar elementos a los datos de train, test '
                                 'y dev. Modo pasado: {i}'.format(i=list_name))

    def custom_temporal_crop(self, video_path, video_index, list_name, custom_fn, method_flag):
        """Metodo que se encarga de realizar el corte temporal customizado en
            un video dado por su path y se agregara a la lista indicada.
                Args:
                    video_path: String del ath donde se encuentran los frames del video.
                    video_index: Entero que corresponde al indice en el df del video, solo se usa
                                         cuando esta method_flag en 2.
                    list_name: String con las opciones ("train", "test", "dev") para escoger
                                      a que lista se agregaran los videos de formas custom.
                    custom_fn: Callback que corresponde a la funcion customizada a la
                                        que le aplicara sobre cada video.
                    method_flag: Variable numerica que me indica en que forma estoy cargando los datos.
        """
        if not isinstance(video_path, str):
            raise ValueError('Se ha pasado video_path como una instancia diferente'
                             'a lo que es un string. Instancia pasada: {i}'.format(i=type(video_path)))
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
                    raise ValueError('Se ha pasado un modo invalido al cual no se le '
                                     'pueden asignar el label del  elemento a los datos de train, test '
                                     'y dev. Modo pasado: {i}'.format(i=list_name))
            else:
                raise ValueError(
                    'El valor pasado a la funcion generate_classes con el parametro method_flag debe ser un numero '
                    'entre 1 y 2. Valor pasado: ' + str(method_flag))
            n_veces = len(frames)
            for i in range(n_veces):
                if len(frames[i]) != self.video_frames:
                    raise ValueError(
                        'La longitud de los frames a agregar por medio de una '
                        'funcion customizada debe ser igual a la especificada '
                        'por el usuario. Longitud encontrada de %s' % len(frames[i])
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
                    raise ValueError('Se ha pasado un modo invalido al cual no se le '
                                     'pueden agregar elementos a los datos de train, test '
                                     'y dev. Modo pasado: {i}'.format(i=list_name))

        except:
            raise AttributeError(
                'Se espera que la funcion customizada retorne una matriz'
                ' donde cada fila corresponde a un video con el corte temporal '
                'y la dimension de columnas sea igual a la longitud de frames'
                ' especificada'
            )

    def none_temporal_crop(self, video_path, video_index, list_name, method_flag):
        """Metodo que se encarga de realizar el corte temporal None en
        un video dado por su path y se agregara a la lista indicada.
        Args:
            video_path: String del ath donde se encuentran los frames del video.
            video_index: Entero que corresponde al indice en el df del video, solo se usa
                                         cuando esta method_flag en 2.
            list_name: String con las opciones ("train", "test", "dev") para escoger
                              a que lista se agregaran los videos de formas None.
            method_flag: Variable numerica que me indica en que forma estoy cargando los datos.
        """
        if not isinstance(video_path, str):
            raise ValueError('Se ha pasado video_path como una instancia diferente'
                             'a lo que es un string. Instancia pasada: {i}'.format(i=type(video_path)))
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
                raise ValueError('Se ha pasado un modo invalido al cual no se le '
                                 'pueden asignar el label del  elemento a los datos de train, test '
                                 'y dev. Modo pasado: {i}'.format(i=list_name))
        else:
            raise ValueError(
                'El valor pasado a la funcion generate_classes con el parametro method_flag debe ser un numero '
                'entre 1 y 2. Valor pasado: ' + str(method_flag))
        elemento = {(name, None): (frames_path, label)}
        self.transformation_index += 1
        if list_name == 'train':
            self.train_data.append(elemento)
        elif list_name == 'test':
            self.test_data.append(elemento)
        elif list_name == 'dev':
            self.dev_data.append(elemento)
        else:
            raise ValueError('Se ha pasado un modo invalido al cual no se le '
                             'pueden agregar elementos a los datos de train, test '
                             'y dev. Modo pasado: {i}'.format(i=list_name))

    def temporal_crop(self, mode , custom_fn, method_flag):
        """Metodo que se encarga de realizar el corte temporal en los videos de
        train, test y dev segun el modo especificado y los agrega a la lista de datos.
        Args:
            mode: String o None que corresponde al modo de aumento de datos.
            custom_fn: Callback o funcion de python que retorna la lista de los path a cargar,
            method_flag: Variable numerica que me indica en que forma estoy cargando los datos.
            """
        if mode == 'sequential':
            """ Modo secuencial, donde se toman todos los frames del video en forma
            secuencial hasta donde el video lo permita"""
            for index, video in enumerate(self.videos_train_path):
                self.sequential_temporal_crop(video, index, "train", method_flag)

            for index, video in enumerate(self.videos_test_path):
                self.sequential_temporal_crop(video,index, "test", method_flag)

            if self.dev_path:
                for index, video in enumerate(self.videos_dev_path):
                    self.sequential_temporal_crop(video,index,"dev",method_flag)

        elif mode == 'random':
            """Modo aleatorio, donde la funcion personalizada corresponde al numero
             de cortes aleatorio por video que se vayan a hacer. Estos cortes mantienen
             la secuencia temporal pero puede tomar el inicio desde la cola y terminar en
             el inicio de los videos."""
            if isinstance(custom_fn, int):
                n_veces = custom_fn
            else:
                raise ValueError(
                    'Al usar el modo de corte temporal aleatorio, custom_fn debe ser un entero'
                    ', el valor entregado es de tipo: %s' % type(custom_fn)
                )
            for index, video in enumerate(self.videos_train_path):
                self.random_temporal_crop(video, index, "train", n_veces, method_flag)

            for index, video in enumerate(self.videos_test_path):
                self.random_temporal_crop(video, index, "test", n_veces, method_flag)

            if self.dev_path:
                for index, video in enumerate(self.videos_dev_path):
                    self.random_temporal_crop(video, index, "dev", n_veces, method_flag)

        elif mode == 'custom':
            """Metodo que se encarga de ejecutar la funcion customizada a cada video
            y ejecutar el metodo para obtener los datos a agregar."""
            if custom_fn:
                for index, video in enumerate(self.videos_train_path):
                    self.custom_temporal_crop(video,index, "train",custom_fn, method_flag)

                for index, video in enumerate(self.videos_test_path):
                    self.custom_temporal_crop(video, index, "test", custom_fn, method_flag)

                if self.dev_path:
                    for index, video in enumerate(self.videos_dev_path):
                        self.custom_temporal_crop(video, index, "dev", custom_fn, method_flag)
            else:
                raise ValueError('Debe pasar la funcion customizada para el '
                    'modo customizado, de lo contrario no podra usarlo. Tipo de dato'
                     ' de funcion customizada recibida: %s' % type(custom_fn))

        else:
            """Modo None, donde se toman los primeros frames del video"""
            for index, video in enumerate(self.videos_train_path):
                self.none_temporal_crop(video, index, "train", method_flag)

            for index, video in enumerate(self.videos_test_path):
                self.none_temporal_crop(video, index, "test", method_flag)

            if self.dev_path:
                for index, video in enumerate(self.videos_dev_path):
                    self.none_temporal_crop(video, index, "dev", method_flag)

    def sequential_frame_crop(self,list_name, conserve):
        """Metodo que se encarga de realizar el corte espacial secuencial en
        un video dado y se modificara en la lista indicada conservando o no los
        datos ya colocados ahi.
        Args:
            list_name:String con las opciones ("train", "test", "dev") para escoger
                              a que lista se agregaran los videos de formas secuencial.
            conserve: Booleano que determina si conservo los datos ya calculados
                            en la lista.
        """
        if list_name == 'train':
            lista = self.train_data
        elif list_name == 'test':
            lista = self.test_data
        elif list_name == 'dev':
            lista = self.dev_data
        else:
            raise ValueError('Se ha pasado un modo invalido al cual no se le '
                             'pueden agregar elementos a los datos de train, test '
                             'y dev. Modo pasado: {i}'.format(i=list_name))
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
        """Metodo que se encarga de realizar el corte espacial aleatorio en
        un video dado y se modificara en la lista indicada conservando o no los
        datos ya colocados ahi.
        Args:
            list_name:String con las opciones ("train", "test", "dev") para escoger
                              a que lista se agregaran los videos de formas secuencial.
            conserve: Booleano que determina si conservo los datos ya calculados
                            en la lista.
            n_veces: Entero que indica cuantos cortes espaciales aleatorios se
                            deben hace sobre el video.
        """
        if list_name == 'train':
            lista = self.train_data
        elif list_name == 'test':
            lista = self.test_data
        elif list_name == 'dev':
            lista = self.dev_data
        else:
            raise ValueError('Se ha pasado un modo invalido al cual no se le '
                             'pueden agregar elementos a los datos de train, test '
                             'y dev. Modo pasado: {i}'.format(i=list_name))
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
        """Metodo que se encarga de realizar el corte espacial aleatorio en
        un video dado y se modificara en la lista indicada conservando o no los
        datos ya colocados ahi.
        Args:
            list_name:String con las opciones ("train", "test", "dev") para escoger
                              a que lista se agregaran los videos de formas secuencial.
            conserve: Booleano que determina si conservo los datos ya calculados
                            en la lista.
            custom_fn: Callback que corresponde a la funcion customizada a la
                                que le aplicara sobre cada video.
        """
        if list_name == 'train':
            lista = self.train_data
        elif list_name == 'test':
            lista = self.test_data
        elif list_name == 'dev':
            lista = self.dev_data
        else:
            raise ValueError('Se ha pasado un modo invalido al cual no se le '
                             'pueden agregar elementos a los datos de train, test '
                             'y dev. Modo pasado: {i}'.format(i=list_name))
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
                            'El tamaño de los frames debe ser igual al tamaño '
                            'especificado por el usuario. Tamaño encontrado de '
                            '%s' % str(size_frame)
                        )
                    function = lambda frame: frame[corte[2]: corte[3], corte[0]: corte[1]]
                    name = "icrop" + str(self.transformation_index)
                    self.transformation_index += 1
                    elemento = {(name, function): values}
                    new_lista.append(elemento)

            except:
                raise AttributeError(
                    'Se espera que la funcion customizada retorne una matriz '
                    'de forma que las filas es un corte a hacerle a cada video y '
                    'las columnas sean 4 (inicio corte x, fin corte x, inicio corte '
                    'y, fin corte y) exactamente en ese orden.'
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
            raise ValueError('Se ha pasado un modo invalido al cual no se le '
                             'pueden agregar elementos a los datos de train, test '
                             'y dev. Modo pasado: {i}'.format(i=list_name))
        for index in range(len(lista)):
            llave_original = tuple(lista[index].keys())[0]
            llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
            valores = tuple(lista[index].values())[0]
            lista[index] = {llave_nueva: valores}

    def frame_crop(self,mode, custom_fn, conserve_original = False):
        """Metodo que se encarga de realizar el corte de una imagen segun el
        tamaño especificado por el usuario siguiendo el modo elegido. Tambien
        sirve para aplicar transformaciones sobre los frames ya que esta operacion
        se hace uno por uno. Al final agregara las transforamciones a la lista
        de datos reemplazando o inmediatamente despues si se quieren conservar.
        Args:
            mode: String o None que corresponde al modo de aumento de datos.
            custom_fn: Callback o funcion de python que retorna la lista de los path a cargar
            converse_original: Booleano por defecto en False que indica si se agregan o reemplazan
            los valores ya almacenados en la lista de datos.
            """
        if mode == 'sequential':
            """Modo secuencial, donde se toman por cada imagen (desde izq a der)
             y arriba hacia abajo el tamaño indicado porel usuario hasta donde se
             le permita"""
            self.sequential_frame_crop("train", conserve_original)
            self.sequential_frame_crop("test", conserve_original)
            if self.dev_path:
                self.sequential_frame_crop("dev", conserve_original)

        elif mode == 'random':
            """Modo aleatorio, donde la funcion personalizada corresponde al numero 
                 de cortes aleatorio por frame que se vayan a hacer. Estos cortes son totalmente 
                 aleatorios desde el inicio hasta el limite donde se pueden recortar los 
                 frames."""
            if isinstance(custom_fn, int):
                n_veces = custom_fn
            else:
                raise ValueError(
                    'Al usar el modo de cortes de frames aleatorio, custom_fn debe ser un entero'
                    ', el valor entregado es de tipo: %s' % type(custom_fn)
                )
            self.random_frame_crop("train",conserve_original, n_veces)
            self.random_frame_crop("test", conserve_original, n_veces)
            if self.dev_path:
                self.random_frame_crop("dev", conserve_original, n_veces)

        elif mode == 'custom':
            """Metodo que se encarga de ejecutar la funcion customizada de corte 
            a cada frame de cada video."""
            if custom_fn:
                self.custom_frame_crop("train", conserve_original, custom_fn)
                self.custom_frame_crop("test", conserve_original, custom_fn)
                if self.dev_path:
                    self.custom_frame_crop("dev", conserve_original, custom_fn)
            else:
                raise ValueError('Debe pasar la funcion customizada para el '
                    'modo customizado, de lo contrario no podra usarlo. Tipo de dato'
                     ' de funcion customizada recibida: %s' % type(custom_fn))

        else:
            """Modo None, donde simplemente se redimensiona toda la imagen"""
            self.none_frame_crop("train")
            self.none_frame_crop("test")
            if self.dev_path:
                self.none_frame_crop("dev")
