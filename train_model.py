from load_data import loadBinaryCovid19ClassificationData

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.applications import ResNet50V2
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

if __name__ == '__main__':

    # image path declaration
    covid_train_path = 'D:\\COVID-19 Radiography Database\\train\\covid'
    notcovid_train_path = 'D:\\COVID-19 Radiography Database\\train\\notcovid'
    covid_validation_path = 'D:\\COVID-19 Radiography Database\\valid\\covid'
    notcovid_validation_path = 'D:\\COVID-19 Radiography Database\\valid\\notcovid'

    # input image shape declaration
    im_shape = (224, 224)

    # training and validation data loading
    cxr_train, label_train = loadCovid19ClassificationData(covid_train_path,notcovid_train_path, im_shape)
    cxr_val, label_val = loadCovid19ClassificationData(covid_validation_path,notcovid_validation_path, im_shape)

    # data label encoding
    label_encoder = LabelEncoder()
    label_train = label_encoder.fit_transform(label_train)
    label_train = to_categorical(label_train)
    label_val = label_encoder.fit_transform(label_val)
    label_val = to_categorical(label_val)

    # model and hyperparameters definition
    cxr_image_shape = cxr_train[0].shape
    num_classes = 2

    resnet50_model = ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=cxr_image_shape
    )

    new_output_layer = resnet50_model.output
    new_output_layer = GlobalAveragePooling2D()(new_output_layer)
    new_output_layer = Dropout(0.5)(new_output_layer)
    new_output_layer = Dense(num_classes, activation='sigmoid')(new_output_layer)
    resnet50_model = Model(inputs=resnet50_model.input, outputs=new_output_layer)

    resnet50_model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    epochs = 30
    results = resnet50_model.fit(
        cxr_train,
        label_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(cxr_val, label_val),
        verbose=2
    )

    # model and weights serialization
    model_json = resnet50_model.to_json()
    with open("covid19_model.json", "w") as json_file:
        json_file.write(model_json)

    resnet50_model.save_weights("trained_covid19_model_weights_20200923_30epochs.h5")
    print("Saved model to disk")