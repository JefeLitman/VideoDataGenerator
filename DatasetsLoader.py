"""Video Data Generator for Any Video Dataset with Custom Transformations
You can use it in your own and only have two dependencies with opencv and numpy.
"""

import os
import cv2
import numpy as np
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

class VideoDataGenerator():
    """Clase para cargar todos los datos de un Dataset a partir de la ruta
    especificada por el usuario y ademas, agregando transformaciones en
    tiempo real.

    En esta version lee los datos a partir de los frames pero no archivos avi.
    """

    def __init__(self,directory_path,
                 batch_size = 32,
                 original_frame_size = None,
                 frame_size = None,
                 video_frames = 16,
                 temporal_crop = (None, None),
                 frame_crop = (None, None),
                 shuffle = False,
                 conserve_original = False
                 ):
        """Constructor de la clase.
        Args:
            directory_path: String que tiene la ruta absoluta de la ubicacion del
                                       dataset, incluyendo en el path el split. Obligatorio.
            batch_size: Numero que corresponde al tamaño de elementos por batch.
                                Por defecto en 32.
            original_frame_size: Tupla de enteros del tamaño de imagen original a cargar con la estructura (width, height).
                                                Por defecto en None y tomara el tamaño original de las imagenes.
            frame_size: Tupla de enteros del tamaña final de imagen que quedara con la estructura (width, height).
                                 Por defecto en None y tomara el tamaño original de las imagenes.
            video_frames: Numero de frames con los que quedara los batch.
                                    Por defecto 16.
            temporal_crop: Tupla de la forma (Modo, funcion customizada o callback)
                                    de python e indica que tipo de corte temporal se hara sobre los videos.
                                    Por defecto en (None, None).
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

        self.ds_directory = directory_path
        self.batch_size = batch_size
        self.video_frames = video_frames
        self.transformation_index = 0
        self.multiproceso = Pool()
        self.multihilo = ThreadPoolExecutor()

        """Proceso de revisar que los directorios del path esten en la jerarquia correcta"""
        directories = os.listdir(self.ds_directory)
        for i in directories:
            if i.lower() == "train":
                self.train_path = os.path.join(self.ds_directory,i)
                self.train_batch_index = 0
                self.train_data = []
                self.dev_path = None
                self.dev_data = None
            elif i.lower() == "test":
                self.test_path = os.path.join(self.ds_directory,i)
                self.test_batch_index = 0
                self.test_data = []
                self.dev_path = None
                self.dev_data = None
            elif i.lower() == "dev":
                self.dev_path = os.path.join(self.ds_directory,i)
                self.dev_batch_index = 0
                self.dev_data = []
            else:
                raise ValueError(
                    'La organizacion de la carpeta debe seguir la estructura'
                    'de train, test y dev (este ultimo opcional) teniendo'
                    'los mismos nombres sin importar mayus o minus.'
                    'Carpeta del error: %s' % i)

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

        """Proceso de definir el tamaño original de todas las imagenes si no se entrega
        el parametro de original size y establecer el tamaño de los frames"""
        if original_frame_size:
            self.original_size = original_frame_size
        else:
            frames_path = os.path.join(self.videos_train_path[0], sorted(os.listdir(self.videos_train_path[0]))[0])
            self.original_size = self.load_raw_frame(frames_path).shape[1::-1]
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

        """Proceso de generar los datos con o sin transformaciones"""
        self. generate_classes()
        self.generate_videos_paths()
        if conserve_original and temporal_crop[0] not in (None, 'sequential'):
            self.temporal_crop(mode = 'sequential', custom_fn=temporal_crop[1])
        self.temporal_crop(mode = temporal_crop[0], custom_fn=temporal_crop[1])
        self.frame_crop(mode=frame_crop[0], custom_fn=frame_crop[1], conserve_original=conserve_original)

        if shuffle:
            self.shuffle_videos()
        self.complete_batches()

    def generate_classes(self):
        """Metodo que se encarga de generar el numero de clases, los nombres
        de clases, numeros con indices de clase y el diccionario que convierte de clase
        a numero como de numero a clase"""

        self.to_class = sorted(os.listdir(self.train_path)) #Equivale al vector de clases
        self.to_number = dict((name, index) for index,name in enumerate(self.to_class))

    def generate_videos_paths(self):
        """Metodo que se encarga de generar una lista con el path absoluto de todos los videos para train, test y
        dev si llega a aplicar esta carpeta."""
        self.videos_train_path = []
        self.videos_test_path = []
        if self.dev_path:
            self.videos_dev_path = []

        for clase in self.to_class:

            videos_train_path = os.path.join(self.train_path,clase)
            self.videos_train_path += [os.path.join(videos_train_path,i) for i in sorted(os.listdir(videos_train_path))]

            videos_test_path = os.path.join(self.test_path,clase)
            self.videos_test_path += [os.path.join(videos_test_path,i) for i in sorted(os.listdir(videos_test_path))]

            if self.dev_path:
                videos_dev_path = os.path.join(self.dev_path,clase)
                self.videos_dev_path += [os.path.join(videos_dev_path,i) for i in sorted(os.listdir(videos_dev_path))]

    def complete_batches(self):
        self.train_batches = int( len(self.train_data) / self.batch_size)
        residuo = len(self.train_data) % self.batch_size
        if residuo != 0:
            self.train_batches += 1
            self.train_data = np.append(self.train_data, self.train_data[:self.batch_size - residuo])
        
        self.test_batches = int( len(self.test_data) /  self.batch_size)
        residuo = len(self.test_data) % self.batch_size
        if residuo != 0:
            self.test_batches += 1
            self.test_data = np.append(self.test_data, self.test_data[:self.batch_size - residuo])
        
        if self.dev_path:
            self.dev_batches = int( len(self.dev_data) / self.batch_size)
            residuo = len(self.dev_data) % self.batch_size
            if residuo != 0:
                self.dev_batches += 1
                self.dev_data = np.append(self.dev_data, self.dev_data[:self.batch_size - residuo])

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

        return np.asarray(video, dtype=np.float32)

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

        self.test_batch_index += 1

        return np.asarray(batch, dtype=np.float32), np.asarray(labels, dtype=np.int32)

    def get_next_dev_batch(self, n_canales=3):
        """Metodo que se encarga de retornar el siguiente batch o primer batch
                de datos dev si se cumple un epoch.
                Args:
                    n_canales: Numero que corresponde al numero de canales que posee las imagenes. Por defecto en 3.
                    """
        if self.dev_path:
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

            self.test_batch_index += 1

            return np.asarray(batch, dtype=np.float32), np.asarray(labels, dtype=np.int32)
        else:
            raise AttributeError(
                'No se puede llamar a la funcion debido a que en el directorio no se'
                'encuentra la carpeta dev y por ende no se tienen datos en dev'
            )

    def load_raw_frame(self,frame_path, channels = 3):
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
        return cv2.resize(img, tuple(self.original_size))

    def resize_frame(self, image):
        """Metodo que se encarga de redimensionar un frame segun el tamaño
        especificado por el usuario"""
        return cv2.resize(image, tuple(self.frame_size))

    def temporal_crop(self, mode , custom_fn):
        """Metodo que se encarga de realizar el corte temporal en los videos de
        train, test y dev segun el modo especificado y los agrega a la lista de datos.
        Args:
            mode: String o None que corresponde al modo de aumento de datos.
            custom_fn: Callback o funcion de python que retorna la lista de los path a cargar,
            """
        if mode == 'sequential':
            """ Modo secuencial, donde se toman todos los frames del video en forma
            secuencial hasta donde el video lo permita"""
            for video in self.videos_train_path:
                frames_path = [os.path.join(video,frame) for frame in sorted(os.listdir(video))]
                while len(frames_path) < self.video_frames:
                    frames_path += frames_path[:self.video_frames - len(frames_path)]
                label = self.to_number[video.split("/")[-2]]

                n_veces = len(frames_path) // self.video_frames
                for i in range(n_veces):
                    start = self.video_frames * i
                    end = self.video_frames * (i+1)
                    frames = frames_path[start:end]

                    name = "tcrop" + str(self.transformation_index)
                    elemento = { (name, None) : (frames, label) }
                    self.transformation_index += 1
                    self.train_data.append(elemento)

            for video in self.videos_test_path:
                frames_path = [os.path.join(video,frame) for frame in sorted(os.listdir(video))]
                while len(frames_path) < self.video_frames:
                    frames_path += frames_path[:self.video_frames - len(frames_path)]
                label = self.to_number[video.split("/")[-2]]

                n_veces = len(frames_path) // self.video_frames
                for i in range(n_veces):
                    start = self.video_frames * i
                    end = self.video_frames * (i + 1)
                    frames = frames_path[start:end]

                    name = "tcrop" + str(self.transformation_index)
                    elemento = {(name, None): (frames, label)}
                    self.transformation_index += 1
                    self.test_data.append(elemento)

            if self.dev_path:
                for video in self.videos_dev_path:
                    frames_path = [os.path.join(video,frame) for frame in sorted(os.listdir(video))]
                    while len(frames_path) < self.video_frames:
                        frames_path += frames_path[:self.video_frames - len(frames_path)]
                    label = self.to_number[video.split("/")[-2]]

                    n_veces = len(frames_path) // self.video_frames
                    for i in range(n_veces):
                        start = self.video_frames * i
                        end = self.video_frames * (i + 1)
                        frames = frames_path[start:end]

                        name = "tcrop" + str(self.transformation_index)
                        elemento = {(name, None): (frames, label)}
                        self.transformation_index += 1
                        self.dev_data.append(elemento)

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
            for video in self.videos_train_path:
                frames_path = [os.path.join(video,frame) for frame in sorted(os.listdir(video))]
                while len(frames_path) < self.video_frames:
                    frames_path += frames_path[:self.video_frames - len(frames_path)]
                label = self.to_number[video.split("/")[-2]]

                for _ in range(n_veces):
                    start = np.random.randint(0,len(frames_path))
                    if start + self.video_frames > len(frames_path):
                        end = self.video_frames + start - len(frames_path)
                        frames = frames_path[start : ] + frames_path[ : end]
                    else:
                        end = start + self.video_frames
                        frames = frames_path[start : end]

                    name = "tcrop" + str(self.transformation_index)
                    elemento = { (name, None) : (frames, label) }
                    self.transformation_index += 1
                    self.train_data.append(elemento)

            for video in self.videos_test_path:
                frames_path = [os.path.join(video,frame) for frame in sorted(os.listdir(video))]
                while len(frames_path) < self.video_frames:
                    frames_path += frames_path[:self.video_frames - len(frames_path)]
                label = self.to_number[video.split("/")[-2]]

                for _ in range(n_veces):
                    start = np.random.randint(0, len(frames_path))
                    if start + self.video_frames > len(frames_path):
                        end = self.video_frames + start - len(frames_path)
                        frames = frames_path[start:] + frames_path[: end]
                    else:
                        end = start + self.video_frames
                        frames = frames_path[start: end]

                    name = "tcrop" + str(self.transformation_index)
                    elemento = {(name, None): (frames, label)}
                    self.transformation_index += 1
                    self.test_data.append(elemento)

            if self.dev_path:
                for video in self.videos_dev_path:
                    frames_path = [os.path.join(video,frame) for frame in sorted(os.listdir(video))]
                    while len(frames_path) < self.video_frames:
                        frames_path += frames_path[:self.video_frames - len(frames_path)]
                    label = self.to_number[video.split("/")[-2]]

                    for _ in range(n_veces):
                        start = np.random.randint(0, len(frames_path))
                        if start + self.video_frames > len(frames_path):
                            end = self.video_frames + start - len(frames_path)
                            frames = frames_path[start:] + frames_path[: end]
                        else:
                            end = start + self.video_frames
                            frames = frames_path[start: end]

                        name = "tcrop" + str(self.transformation_index)
                        elemento = {(name, None): (frames, label)}
                        self.transformation_index += 1
                        self.dev_data.append(elemento)

        elif mode == 'custom':
            """Metodo que se encarga de ejecutar la funcion customizada a cada video
            y ejecutar el metodo para obtener los datos a agregar."""
            if custom_fn:
                for video in self.videos_train_path:
                    frames_path = [os.path.join(video,frame) for frame in sorted(os.listdir(video))]
                    frames = custom_fn(frames_path)
                    label = self.to_number[video.split("/")[-2]]

                    try:
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
                            self.train_data.append(elemento)

                    except:
                        raise AttributeError(
                            'Se espera que la funcion customizada retorne una matriz'
                            ' donde cada fila corresponde a un video con el corte temporal '
                            'y la dimension de columnas sea igual a la longitud de frames'
                            ' especificada'
                        )

                for video in self.videos_test_path:
                    frames_path = [os.path.join(video,frame) for frame in sorted(os.listdir(video))]
                    frames = custom_fn(frames_path)
                    label = self.to_number[video.split("/")[-2]]

                    try:
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
                            self.test_data.append(elemento)

                    except:
                        raise AttributeError(
                            'Se espera que la funcion customizada retorne una matriz'
                            ' donde cada fila corresponde a un video con el corte temporal '
                            'y la dimension de columnas sea igual a la longitud de frames'
                            ' especificada'
                        )

                if self.dev_path:
                    for video in self.videos_dev_path:
                        frames_path = [os.path.join(video,frame) for frame in sorted(os.listdir(video))]
                        frames = custom_fn(frames_path)
                        label = self.to_number[video.split("/")[-2]]

                        try:
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
                                self.dev_data.append(elemento)

                        except:
                            raise AttributeError(
                                'Se espera que la funcion customizada retorne una matriz'
                                ' donde cada fila corresponde a un video con el corte temporal '
                                'y la dimension de columnas sea igual a la longitud de frames'
                                ' especificada'
                            )
            else:
                raise ValueError('Debe pasar la funcion customizada para el '
                    'modo customizado, de lo contrario no podra usarlo. Tipo de dato'
                     ' de funcion customizada recibida: %s' % type(custom_fn))

        else:
            """Modo None, donde se toman los primeros frames del video"""
            for video in self.videos_train_path:
                name = "tcrop" + str(self.transformation_index)
                frames_path = [os.path.join(video,frame) for frame in sorted(os.listdir(video))]
                while len(frames_path) < self.video_frames:
                    frames_path += frames_path[:self.video_frames - len(frames_path)]
                frames_path = frames_path[:self.video_frames]
                label = self.to_number[video.split("/")[-2]]
                elemento = { (name, None) : (frames_path, label) }
                self.transformation_index += 1
                self.train_data.append(elemento)

            for video in self.videos_test_path:
                name = "tcrop" + str(self.transformation_index)
                frames_path = [os.path.join(video,frame) for frame in sorted(os.listdir(video))]
                while len(frames_path) < self.video_frames:
                    frames_path += frames_path[:self.video_frames - len(frames_path)]
                frames_path = frames_path[:self.video_frames]
                label = self.to_number[video.split("/")[-2]]
                elemento = {(name, None): (frames_path, label)}
                self.transformation_index += 1
                self.test_data.append(elemento)

            if self.dev_path:
                for video in self.videos_dev_path:
                    name = "tcrop" + str(self.transformation_index)
                    frames_path = [os.path.join(video,frame) for frame in sorted(os.listdir(video))]
                    while len(frames_path) < self.video_frames:
                        frames_path += frames_path[:self.video_frames - len(frames_path)]
                    frames_path = frames_path[:self.video_frames]
                    label = self.to_number[video.split("/")[-2]]
                    elemento = {(name, None): (frames_path, label)}
                    self.transformation_index += 1
                    self.dev_data.append(elemento)

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
        original_height = self.original_size[1]
        original_width = self.original_size[0]
        if mode == 'sequential':
            """Modo secuencial, donde se toman por cada imagen (desde izq a der)
             y arriba hacia abajo el tamaño indicado porel usuario hasta donde se
             le permita"""
            if conserve_original:
                n = len(self.train_data)
                for index in range(n):
                    #Agrego los nuevos cortes de frames a los datos
                    values = tuple(self.train_data[index].values())[0]
                    n_veces = [original_width // self.frame_size[0], original_height//self.frame_size[1]]

                    for i in range(n_veces[0]):
                        for j in range(n_veces[1]):
                            start_width = i * self.frame_size[0]
                            end_width = start_width + self.frame_size[0]
                            start_height = j * self.frame_size[1]
                            end_height = start_height + self.frame_size[1]
                            function = lambda frame: frame[start_height : end_height, start_width : end_width]

                            name = "icrop" + str(self.transformation_index)
                            self.transformation_index += 1
                            elemento = { (name, function) : values }
                            self.train_data.append(elemento)

                    #Reemplazo la funcion de los datos ya almacenados
                    llave_original = tuple(self.train_data[index].keys())[0]
                    llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                    valores = tuple(self.train_data[index].values())[0]
                    self.train_data[index] = {llave_nueva: valores}

                n = len(self.test_data)
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los datos
                    values = tuple(self.test_data[index].values())[0]
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
                            self.test_data.append(elemento)

                    # Reemplazo la funcion de los datos ya almacenados
                    llave_original = tuple(self.test_data[index].keys())[0]
                    llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                    valores = tuple(self.test_data[index].values())[0]
                    self.test_data[index] = {llave_nueva: valores}

                if self.dev_path:
                    n = len(self.dev_data)
                    for index in range(n):
                        # Agrego los nuevos cortes de frames a los datos
                        values = tuple(self.dev_data[index].values())[0]
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
                                self.dev_data.append(elemento)

                        # Reemplazo la funcion de los datos ya almacenados
                        llave_original = tuple(self.dev_data[index].keys())[0]
                        llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                        valores = tuple(self.dev_data[index].values())[0]
                        self.dev_data[index] = {llave_nueva: valores}
            else:
                n = len(self.train_data)
                new_train_data = []
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los nuevos datos
                    values = tuple(self.train_data[index].values())[0]
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
                            new_train_data.append(elemento)
                self.train_data = new_train_data

                n = len(self.test_data)
                new_test_data = []
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los nuevos datos
                    values = tuple(self.test_data[index].values())[0]
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
                            new_test_data.append(elemento)
                self.test_data = new_test_data

                if self.dev_path:
                    n = len(self.dev_data)
                    new_dev_data = []
                    for index in range(n):
                        # Agrego los nuevos cortes de frames a los nuevos datos
                        values = tuple(self.dev_data[index].values())[0]
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
                                new_dev_data.append(elemento)
                    self.dev_data = new_dev_data

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
            if conserve_original:
                n = len(self.train_data)
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los datos
                    values = tuple(self.train_data[index].values())[0]

                    for _ in range(n_veces):
                        start_width = np.random.randint(0, original_width - self.frame_size[0])
                        end_width = start_width + self.frame_size[0]
                        start_height =np.random.randint(0, original_height - self.frame_size[1])
                        end_height = start_height + self.frame_size[1]
                        function = lambda frame: frame[start_height: end_height, start_width: end_width]

                        name = "icrop" + str(self.transformation_index)
                        self.transformation_index += 1
                        elemento = {(name, function): values}
                        self.train_data.append(elemento)

                    # Reemplazo la funcion de los datos ya almacenados
                    llave_original = tuple(self.train_data[index].keys())[0]
                    llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                    valores = tuple(self.train_data[index].values())[0]
                    self.train_data[index] = {llave_nueva: valores}

                n = len(self.test_data)
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los datos
                    values = tuple(self.test_data[index].values())[0]

                    for _ in range(n_veces):
                        start_width = np.random.randint(0, original_width - self.frame_size[0])
                        end_width = start_width + self.frame_size[0]
                        start_height = np.random.randint(0, original_height - self.frame_size[1])
                        end_height = start_height + self.frame_size[1]
                        function = lambda frame: frame[start_height: end_height, start_width: end_width]

                        name = "icrop" + str(self.transformation_index)
                        self.transformation_index += 1
                        elemento = {(name, function): values}
                        self.test_data.append(elemento)

                    # Reemplazo la funcion de los datos ya almacenados
                    llave_original = tuple(self.test_data[index].keys())[0]
                    llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                    valores = tuple(self.test_data[index].values())[0]
                    self.test_data[index] = {llave_nueva: valores}

                if self.dev_path:
                    n = len(self.dev_data)
                    for index in range(n):
                        # Agrego los nuevos cortes de frames a los datos
                        values = tuple(self.dev_data[index].values())[0]

                        for _ in range(n_veces):
                            start_width = np.random.randint(0, original_width - self.frame_size[0])
                            end_width = start_width + self.frame_size[0]
                            start_height = np.random.randint(0, original_height - self.frame_size[1])
                            end_height = start_height + self.frame_size[1]
                            function = lambda frame: frame[start_height: end_height, start_width: end_width]

                            name = "icrop" + str(self.transformation_index)
                            self.transformation_index += 1
                            elemento = {(name, function): values}
                            self.dev_data.append(elemento)

                        # Reemplazo la funcion de los datos ya almacenados
                        llave_original = tuple(self.dev_data[index].keys())[0]
                        llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                        valores = tuple(self.dev_data[index].values())[0]
                        self.dev_data[index] = {llave_nueva: valores}
            else:
                n = len(self.train_data)
                new_train_data = []
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los datos
                    values = tuple(self.train_data[index].values())[0]

                    for _ in range(n_veces):
                        start_width = np.random.randint(0, original_width - self.frame_size[0])
                        end_width = start_width + self.frame_size[0]
                        start_height = np.random.randint(0, original_height - self.frame_size[1])
                        end_height = start_height + self.frame_size[1]
                        function = lambda frame: frame[start_height: end_height, start_width: end_width]

                        name = "icrop" + str(self.transformation_index)
                        self.transformation_index += 1
                        elemento = {(name, function): values}
                        new_train_data.append(elemento)
                self.train_data = new_train_data

                n = len(self.test_data)
                new_test_data = []
                for index in range(n):
                    # Agrego los nuevos cortes de frames a los datos
                    values = tuple(self.test_data[index].values())[0]

                    for _ in range(n_veces):
                        start_width = np.random.randint(0, original_width - self.frame_size[0])
                        end_width = start_width + self.frame_size[0]
                        start_height = np.random.randint(0, original_height - self.frame_size[1])
                        end_height = start_height + self.frame_size[1]
                        function = lambda frame: frame[start_height: end_height, start_width: end_width]

                        name = "icrop" + str(self.transformation_index)
                        self.transformation_index += 1
                        elemento = {(name, function): values}
                        new_test_data.append(elemento)
                self.test_data = new_test_data

                if self.dev_path:
                    n = len(self.dev_data)
                    new_dev_data = []
                    for index in range(n):
                        # Agrego los nuevos cortes de frames a los datos
                        values = tuple(self.dev_data[index].values())[0]

                        for _ in range(n_veces):
                            start_width = np.random.randint(0, original_width - self.frame_size[0])
                            end_width = start_width + self.frame_size[0]
                            start_height = np.random.randint(0, original_height - self.frame_size[1])
                            end_height = start_height + self.frame_size[1]
                            function = lambda frame: frame[start_height: end_height, start_width: end_width]

                            name = "icrop" + str(self.transformation_index)
                            self.transformation_index += 1
                            elemento = {(name, function): values}
                            new_dev_data.append(elemento)
                    self.dev_data = new_dev_data

        elif mode == 'custom':
            """Metodo que se encarga de ejecutar la funcion customizada de corte 
            a cada frame de cada video."""
            if custom_fn:
                if conserve_original:
                    n = len(self.train_data)
                    for index in range(n):
                        #Arego lso nuevos cortes de franes a los datos
                        values = tuple(self.train_data[index].values())[0]
                        cortes = custom_fn(original_width, original_height)

                        try:
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
                                elemento = { (name, function) : values }
                                self.train_data.append(elemento)

                        except:
                            raise AttributeError(
                                'Se espera que la funcion customizada retorne una matriz '
                                'de forma que las filas es un corte a hacerle a cada video y '
                                'las columnas sean 4 (inicio corte x, fin corte x, inicio corte '
                                'y, fin corte y) exactamente en ese orden.'
                            )
                        # Reemplazo la funcion de los datos ya almacenados
                        llave_original = tuple(self.train_data[index].keys())[0]
                        llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                        self.train_data[index] = {llave_nueva: values}

                    n = len(self.test_data)
                    for index in range(n):
                        # Arego lso nuevos cortes de franes a los datos
                        values = tuple(self.test_data[index].values())[0]
                        cortes = custom_fn(original_width, original_height)

                        try:
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
                                self.test_data.append(elemento)

                        except:
                            raise AttributeError(
                                'Se espera que la funcion customizada retorne una matriz '
                                'de forma que las filas es un corte a hacerle a cada video y '
                                'las columnas sean 4 (inicio corte x, fin corte x, inicio corte '
                                'y, fin corte y) exactamente en ese orden.'
                            )
                        # Reemplazo la funcion de los datos ya almacenados
                        llave_original = tuple(self.test_data[index].keys())[0]
                        llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                        self.test_data[index] = {llave_nueva: values}

                    if self.dev_path:
                        n = len(self.dev_data)
                        for index in range(n):
                            # Arego lso nuevos cortes de franes a los datos
                            values = tuple(self.dev_data[index].values())[0]
                            cortes = custom_fn(original_width, original_height)

                            try:
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
                                    self.dev_data.append(elemento)

                            except:
                                raise AttributeError(
                                    'Se espera que la funcion customizada retorne una matriz '
                                    'de forma que las filas es un corte a hacerle a cada video y '
                                    'las columnas sean 4 (inicio corte x, fin corte x, inicio corte '
                                    'y, fin corte y) exactamente en ese orden.'
                                )
                            # Reemplazo la funcion de los datos ya almacenados
                            llave_original = tuple(self.dev_data[index].keys())[0]
                            llave_nueva = (
                            llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                            self.dev_data[index] = {llave_nueva: values}

                else:
                    n = len(self.train_data)
                    new_train_data = []
                    for index in range(n):
                        # Reemplazo los nuevos cortes de frames a los datos
                        values = tuple(self.train_data[index].values())[0]
                        cortes = custom_fn(original_width, original_height)

                        try:
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
                                new_train_data.append(elemento)

                        except:
                            raise AttributeError(
                                'Se espera que la funcion customizada retorne una matriz '
                                'de forma que las filas es un corte a hacerle a cada video y '
                                'las columnas sean 4 (inicio corte x, fin corte x, inicio corte '
                                'y, fin corte y) exactamente en ese orden.'
                            )
                    self.train_data = new_train_data

                    n = len(self.test_data)
                    new_test_data = []
                    for index in range(n):
                        # Reemplazo los nuevos cortes de frames a los datos
                        values = tuple(self.test_data[index].values())[0]
                        cortes = custom_fn(original_width, original_height)

                        try:
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
                                new_test_data.append(elemento)

                        except:
                            raise AttributeError(
                                'Se espera que la funcion customizada retorne una matriz '
                                'de forma que las filas es un corte a hacerle a cada video y '
                                'las columnas sean 4 (inicio corte x, fin corte x, inicio corte '
                                'y, fin corte y) exactamente en ese orden.'
                            )
                    self.test_data = new_test_data

                    if self.dev_path:
                        n = len(self.dev_data)
                        new_dev_data = []
                        for index in range(n):
                            # Reemplazo los nuevos cortes de frames a los datos
                            values = tuple(self.dev_data[index].values())[0]
                            cortes = custom_fn(original_width, original_height)

                            try:
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
                                    new_dev_data.append(elemento)

                            except:
                                raise AttributeError(
                                    'Se espera que la funcion customizada retorne una matriz '
                                    'de forma que las filas es un corte a hacerle a cada video y '
                                    'las columnas sean 4 (inicio corte x, fin corte x, inicio corte '
                                    'y, fin corte y) exactamente en ese orden.'
                                )
                        self.dev_data = new_dev_data

            else:
                raise ValueError('Debe pasar la funcion customizada para el '
                    'modo customizado, de lo contrario no podra usarlo. Tipo de dato'
                     ' de funcion customizada recibida: %s' % type(custom_fn))

        else:
            """Modo None, donde simplemente se redimensiona toda la imagen"""
            for index in range(len(self.train_data)):
                llave_original = tuple(self.train_data[index].keys())[0]
                llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                valores = tuple(self.train_data[index].values())[0]
                self.train_data[index] = {llave_nueva: valores}

            for index in range(len(self.test_data)):
                llave_original = tuple(self.test_data[index].keys())[0]
                llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                valores = tuple(self.test_data[index].values())[0]
                self.test_data[index] = {llave_nueva: valores}

            if self.dev_path:
                for index in range(len(self.dev_data)):
                    llave_original = tuple(self.dev_data[index].keys())[0]
                    llave_nueva = (llave_original[0] + "icrop" + str(self.transformation_index), self.resize_frame)
                    valores = tuple(self.dev_data[index].values())[0]
                    self.dev_data[index] = {llave_nueva: valores}
