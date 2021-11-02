import tensorflow as tf
from model import ViT
import tensorflow_addons as tfa

from config import Config



def main():
    conf = Config()
    model = ViT(
        **conf.model
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        optimizer=tfa.optimizers.AdamW(
            **conf.optimizer
        ),
        metrics=["accuracy"],
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=conf.log_dir, histogram_freq=1)
    x_train, y_train, x_test, y_test = load_dataset()
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=conf.batch_size,
        epochs=conf.epochs,
        validation_split=0.1,
        callbacks=[tensorboard_callback],
    )
    model.save(conf.save_dir)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")


def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    return x_train, y_train, x_test, y_test

if __name__=="__main__":
    main()