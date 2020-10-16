import datetime
import os

import tensorflow as tf
import tensorflow_cloud as tfc
import tensorflow_datasets as tfds

GCP_BUCKET = "your-bucket-here"


def get_data(img_size, batch_size):
    (ds_train, ds_test), metadata = tfds.load(
        "stanford_dogs",
        split=["train", "test"],
        shuffle_files=True,
        with_info=True,
        as_supervised=True,
    )

    num_classes = metadata.features["label"].num_classes
    size = (img_size, img_size)
    ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
    ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))
    ds_train = ds_train.map(
        input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(input_preprocess)
    ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

    return ds_train, ds_test, num_classes


def input_preprocess(image, label):
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label


def get_model(img_size, num_classes):
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=inputs
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs, outputs)
    base_model.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def get_callbacks(model_path, gcp_bucket):
    checkpoint_path = os.path.join("gs://", gcp_bucket, model_path, "save_at_{epoch}")
    tensorboard_path = os.path.join(
        "gs://", gcp_bucket, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
    ]
    return callbacks


def run():
    model_path = "resnet-dogs"
    img_size = 224
    batch_size = 64
    train_data, test_data, num_classes = get_data(img_size, batch_size)
    model = get_model(img_size, num_classes)
    callbacks = get_callbacks(model_path, GCP_BUCKET)
    if tfc.remote():
        epochs = 500
        model.fit(
            train_data, epochs=epochs, callbacks=callbacks, validation_data=test_data, verbose=2
        )
        save_path = os.path.join("gs://", GCP_BUCKET, model_path)
        model.save(save_path)

        model = tf.keras.models.load_model(save_path)
        model.evaluate(test_data)

    tfc.run(
        requirements_txt="requirements.txt",
        distribution_strategy="auto",
        chief_config=tfc.MachineConfig(
            cpu_cores=8,
            memory=30,
            accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_T4,
            accelerator_count=2,
        ),
        docker_image_bucket_name=GCP_BUCKET,
    )

if __name__ == '__main__':
    run()