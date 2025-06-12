# -*- coding: utf-8 -*-
# author: haroldchen0414

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import pickle
import random
import os
import cv2

class ExtraFeaturesVGG16():
    def __init__(self):
        self.baseDatasetPath = os.path.sep.join(["E:", "dataset"])
        self.dataset = "Food-5k_rebuild"
        self.batchSize = 32
        self.datasetSplits = ["training", "evaluation", "validation"]
        self.classes = ["non-food", "food"]
        self.le = None

    def extra_features(self):
        model = VGG16(weights="imagenet", include_top=False)

        for split in self.datasetSplits:
            print("正在处理{}".format(split))

            p = os.path.sep.join([self.baseDatasetPath, self.dataset, split])
            imagePaths = list(paths.list_images(p))
            random.shuffle(imagePaths)
            labels = [p.split(os.path.sep)[-2] for p in imagePaths]

            if self.le is None:
                le = LabelEncoder()
                le.fit(labels)
            
            csvPath = os.path.sep.join(["output", "vgg16_{}.csv".format(split)])
            
            with open(csvPath, "w") as f:
                for (batch, i) in enumerate(range(0, len(imagePaths), self.batchSize)):
                    print("正在处理批次{}/{}".format(batch + 1, int(np.ceil(len(imagePaths) / float(self.batchSize)))))
                    batchPaths = imagePaths[i: i + self.batchSize]
                    batchLabels = le.transform(labels[i: i + self.batchSize])
                    batchImages = []

                    for imagePath in batchPaths:
                        image = load_img(imagePath, target_size=(224, 224))
                        image = img_to_array(image)

                        image = np.expand_dims(image, axis=0)
                        image = vgg16_preprocess_input(image)
                        batchImages.append(image)
                    
                    batchImages = np.vstack(batchImages)
                    features = model.predict(batchImages, batch_size=self.batchSize)
                    features = features.reshape(features.shape[0], 7 * 7 * 512)

                    for (label, vec) in zip(batchLabels, features):
                        vec = ",".join([str(v) for v in vec])
                        f.write("{},{}\n".format(label, vec))

        with open(os.path.sep.join(["output", "vgg16_le.cpickle"]), "wb") as f:
            f.write(pickle.dumps(le))

    def load_csv_data_split(self, csv_path):
        data = []
        labels = []

        for row in open(csv_path, "rt"):
            row = row.strip().split(",")
            label = row[0]
            features = np.array(row[1:], dtype="float")
            data.append(features)
            labels.append(label)

        data = np.array(data)
        labels = np.array(labels)

        return (data, labels)
    
    def train(self):
        self.extra_features()

        trainPath = os.path.sep.join(["output", "vgg16_training.csv"])
        testPath = os.path.sep.join(["output", "vgg16_evaluation.csv"])

        (trainX, trainY) = self.load_csv_data_split(trainPath)
        (testX, testY) = self.load_csv_data_split(testPath)

        le = pickle.loads(open(os.path.sep.join(["output", "vgg16_le.cpickle"]), "rb").read())

        model = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150)
        model.fit(trainX, trainY)

        preds = model.predict(testX)
        print(classification_report(testY, preds, target_names=le.classes_))

        with open(os.path.sep.join(["output", "vgg16_model.cpickle"]), "wb") as f:
            f.write(pickle.dumps(model))

    def predict(self, image_path=None, csv_path=None):
        modelPath = os.path.sep.join(["output", "vgg16_model.cpickle"])
        lePath = os.path.sep.join(["output", "vgg16_le.cpickle"])
        model = pickle.loads(open(modelPath, "rb").read())
        le = pickle.loads(open(lePath, "rb").read())

        if csv_path:
            (data, labels) = self.load_csv_data_split(csv_path)
            preds = model.predict(data)
            print(classification_report(labels, preds, target_names=le.classes_))

        elif image_path:
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = vgg16_preprocess_input(image)

            baseModel = VGG16(weights="imagenet", include_top=False)
            features = baseModel.predict(image)
            features = features.reshape(features.shape[0], 7 * 7 * 512)

            probs = model.predict_proba(features)[0]
            predClass = le.inverse_transform([np.argmax(probs)])[0]
            predProb = probs[np.argmax(probs)]

            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            text1 = "Class: {}".format(predClass)
            text2 = "Prob: {:.0%}".format(predProb)
            cv2.putText(image, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)        
            cv2.imshow("Image", image)
            cv2.waitKey(0)     

class ExtraFeaturesResNet50():
    def __init__(self):
        self.baseDatasetPath = os.path.sep.join(["E:", "dataset"])
        self.dataset = "Food-5k_rebuild"
        self.batchSize = 32
        self.datasetSplits = ["training", "evaluation", "validation"]
        self.classes = ["non-food", "food"]
        self.le = None

    def extra_features(self):
        model = ResNet50(weights="imagenet", include_top=False)

        for split in self.datasetSplits:
            print("正在处理{}".format(split))

            p = os.path.sep.join([self.baseDatasetPath, self.dataset, split])
            imagePaths = list(paths.list_images(p))
            random.shuffle(imagePaths)
            labels = [p.split(os.path.sep)[-2] for p in imagePaths]

            if self.le is None:
                le = LabelEncoder()
                le.fit(labels)
            
            csvPath = os.path.sep.join(["output", "resnet50_{}.csv".format(split)])
            
            with open(csvPath, "w") as f:
                for (batch, i) in enumerate(range(0, len(imagePaths), self.batchSize)):
                    print("正在处理批次{}/{}".format(batch + 1, int(np.ceil(len(imagePaths) / float(self.batchSize)))))
                    batchPaths = imagePaths[i: i + self.batchSize]
                    batchLabels = le.transform(labels[i: i + self.batchSize])
                    batchImages = []

                    for imagePath in batchPaths:
                        image = load_img(imagePath, target_size=(224, 224))
                        image = img_to_array(image)

                        image = np.expand_dims(image, axis=0)
                        image = resnet50_preprocess_input(image)
                        batchImages.append(image)
                    
                    batchImages = np.vstack(batchImages)
                    features = model.predict(batchImages, batch_size=self.batchSize)
                    features = features.reshape(features.shape[0], 7 * 7 * 2048)

                    for (label, vec) in zip(batchLabels, features):
                        vec = ",".join([str(v) for v in vec])
                        f.write("{},{}\n".format(label, vec))

        with open(os.path.sep.join(["output", "resnet50_le.cpickle"]), "wb") as f:
            f.write(pickle.dumps(le))

    def load_csv_data_split(self, csv_path):
        data = []
        labels = []

        for row in open(csv_path, "rt"):
            row = row.strip().split(",")
            label = row[0]
            features = np.array(row[1:], dtype="float")
            data.append(features)
            labels.append(label)

        data = np.array(data)
        labels = np.array(labels)

        return (data, labels)
    
    def csv_feature_generator(self, csv_path, batch_size, num_classes, mode="train"):
        f = open(csv_path, "r")

        while True:
            data = []
            labels = []

            while len(data) < batch_size:
                row = f.readline()

                if row == "":
                    f.seek(0)
                    row = f.readline()

                    if mode == "eval":
                        break

                row = row.strip().split(",")
                label = row[0]
                label = to_categorical(label, num_classes=num_classes)
                features = np.array(row[1:], dtype="float")
                data.append(features)
                labels.append(label)
            
            yield (np.array(data), np.array(labels))

    def train(self):
        self.extra_features()

        trainPath = os.path.sep.join(["output", "resnet50_training.csv"])
        testPath = os.path.sep.join(["output", "resnet50_evaluation.csv"])
        valPath = os.path.sep.join(["output", "resnet50_validation.csv"])

        totalTrain = sum([1 for i in open(trainPath)])
        testLabels = [int(row.split(",")[0]) for row in open(testPath)]
        totalTest = len(testLabels)
        totalVal = sum([1 for i in open(valPath)])        

        le = pickle.loads(open(os.path.sep.join(["output", "resnet50_le.cpickle"]), "rb").read())

        trainGen = self.csv_feature_generator(trainPath, self.batchSize, len(self.classes), mode="train")
        testGen = self.csv_feature_generator(testPath, self.batchSize, len(self.classes), mode="eval")
        valGen = self.csv_feature_generator(valPath, self.batchSize, len(self.classes), mode="eval")        

        model = Sequential()
        model.add(Dense(256, input_shape=(7 * 7 * 2048, ), activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(len(self.classes), activation="softmax"))        

        # 配置优化器
        opt = SGD(learning_rate=1e-3, momentum=0.9, decay=1e-3 / 25)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        # 配置回调函数
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ModelCheckpoint(filepath=os.path.sep.join(["output", "best_model.h5"]), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        ]
        H = model.fit(x=trainGen, steps_per_epoch=totalTrain // self.batchSize, validation_data=valGen, validation_steps=totalVal // self.batchSize, epochs=25, callbacks=callbacks)

        predIdxs = model.predict(x=testGen, steps=(totalTest // self.batchSize) + 1)
        predIdxs = np.argmax(predIdxs, axis=1)
        print(classification_report(testLabels, predIdxs, target_names=le.classes_))
        model.save(os.path.sep.join(["output", "resnet50_model.h5"]))

    def predict(self, image_path=None, csv_path=None):
        modelPath = os.path.sep.join(["output", "resnet50_model.h5"])
        lePath = os.path.sep.join(["output", "resnet50_le.cpickle"])
        model = load_model(modelPath)
        le = pickle.loads(open(lePath, "rb").read())

        if csv_path:
            (data, rawLabels) = self.load_csv_data_split(csv_path)
            labels = np.array([int(label) for label in rawLabels])

            preds = model.predict(data)
            preds = np.argmax(preds, axis=1)

            print(classification_report(labels, preds, target_names=le.classes_))

        elif image_path:
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = resnet50_preprocess_input(image)

            baseModel = ResNet50(weights="imagenet", include_top=False)
            features = baseModel.predict(image)
            features = features.reshape(features.shape[0], 7 * 7 * 2048)

            probs = model.predict(features)
            predClass = le.inverse_transform([np.argmax(probs[0])])[0]
            predProb = probs[0][np.argmax(probs[0])]

            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            text1 = "Class: {}".format(predClass)
            text2 = "Prob: {:.0%}".format(predProb)
            cv2.putText(image, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)        
            cv2.imshow("Image", image)
            cv2.waitKey(0)  

class FineTuningVGG16():
    def __init__(self):
        self.classes = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]
        self.batchSize = 32

    def plot_training(self, H, plotPath):
        plt.style.use("ggplot")
        plt.figure()
        N = len(H.history["loss"])
        
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(plotPath)
 
    def train(self):
        trainPath = os.path.sep.join(["E:", "dataset", "Food-11", "training"])
        testPath = os.path.sep.join(["E:", "dataset", "Food-11", "evaluation"])
        valPath = os.path.sep.join(["E:", "dataset", "Food-11", "validation"])

        totalTrain = len(list(paths.list_images(trainPath)))
        totalTest = len(list(paths.list_images(testPath)))
        totalVal = len(list(paths.list_images(valPath)))

        trainAug = ImageDataGenerator(rotation_range=30,
                                      zoom_range=0.15,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.15,
                                      horizontal_flip=True,
                                      fill_mode="nearest")
        valAug = ImageDataGenerator()
        # imagenetBGR三通道的全局像素均值
        mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        trainAug.mean = mean
        valAug.mean = mean

        trainGen = trainAug.flow_from_directory(trainPath,
                                            class_mode="categorical",
                                            target_size=(224, 224),
                                            color_mode="rgb",
                                            shuffle=True,
                                            batch_size=self.batchSize)
        testGen = valAug.flow_from_directory(testPath,
                                            class_mode="categorical",
                                            target_size=(224, 224),
                                            color_mode="rgb",
                                            shuffle=False,
                                            batch_size=self.batchSize)
        valGen = valAug.flow_from_directory(valPath,
                                            class_mode="categorical",
                                            target_size=(224, 224),
                                            color_mode="rgb",
                                            shuffle=False,
                                            batch_size=self.batchSize)        

        baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(512, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(len(self.classes), activation="softmax")(headModel)
        model = Model(inputs=baseModel.input, outputs=headModel)

        for layer in baseModel.layers:
            layer.trainable = False
        
        opt = SGD(learning_rate=1e-4, momentum=0.9)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ModelCheckpoint(filepath=os.path.sep.join(["output", "warmup_best_model.h5"]), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        ]

        H = model.fit(x=trainGen, steps_per_epoch=totalTrain // self.batchSize, validation_data=valGen, validation_steps=totalVal // self.batchSize, epochs=50, callbacks=callbacks)
        
        testGen.reset()
        predIdxs = model.predict(x=testGen, steps=(totalTest // self.batchSize) + 1)
        predIdxs = np.argmax(predIdxs, axis=1)

        print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))
        self.plot_training(H, os.path.sep.join(["output", "warmup.png"]))     

        trainGen.reset()
        valGen.reset()

        for layer in baseModel.layers[15:]:
            layer.trainable = True

        for layer in baseModel.layers:
            print("{}: {}".format(layer.name, layer.trainable))
        
        opt = Adam(learning_rate=1e-5)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath=os.path.sep.join(["output", "unfrozen_best_model1.h5"]), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        ]     

        H = model.fit(x=trainGen, steps_per_epoch=totalTrain // self.batchSize, validation_data=valGen, validation_steps=totalVal // self.batchSize, epochs=20, callbacks=callbacks)

        testGen.reset()
        predIdxs = model.predict(x=testGen, steps=(totalTest // self.batchSize) + 1)
        predIdxs = np.argmax(predIdxs, axis=1)
        self.plot_training(H, os.path.sep.join(["output", "unfrozen.png"]))

        model.save(os.path.sep.join(["output", "food11_model"]), save_format="h5")

    def predict(self, image_path):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        output = image.copy()
        output = cv2.resize(output, (400, 400))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype("float32")
        mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")
        image -= mean

        model = load_model(os.path.sep.join(["output", "food11_model"]))
        preds = model.predict(np.expand_dims(image, axis=0))[0]
        i = np.argmax(preds)
        label = self.classes[i]

        text = "{}: {:.0%}".format(label, preds[i])
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Image", output)
        cv2.waitKey(0)

if __name__ == "__main__":
    # VGG16特征提取
    #extraFeaturesVGG16 = ExtraFeaturesVGG16()
    #extraFeaturesVGG16.train()
    #extraFeaturesVGG16.predict(image_path=os.path.sep.join(["E:", "dataset", "Food-5K", "validation", "0_24.jpg"]))
    #extraFeaturesVGG16.predict(csv_path=os.path.sep.join(["output", "vgg16_validation.csv"]))

    #ResNet50特征提取
    #extraFeaturesResNet50 = ExtraFeaturesResNet50()
    #extraFeaturesResNet50.train()
    #extraFeaturesResNet50.predict(image_path=os.path.sep.join(["E:", "dataset", "Food-5K", "validation", "1_224.jpg"]))
    #extraFeaturesResNet50.predict(csv_path=os.path.sep.join(["output", "resnet50_validation.csv"]))

    # VGG16微调
    fineTuningVGG16 = FineTuningVGG16()
    #fineTuningVGG16.train()
    fineTuningVGG16.predict(image_path=os.path.sep.join(["E:", "dataset", "Food-11", "validation", "Meat", "236.jpg"]))