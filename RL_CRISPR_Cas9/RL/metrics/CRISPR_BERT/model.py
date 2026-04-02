import os

import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from keras_bert import load_trained_model_from_checkpoint


# GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_DIR = os.path.join(BASE_DIR, "weight", "bert_weight", "uncased_L-2_H-256_A-4")

CONFIG_PATH = os.path.join(BERT_DIR, "bert_config.json")
CHECKPOINT_PATH = os.path.join(BERT_DIR, "bert_model.ckpt")

# Preload BERT model (non-trainable reference)
bert_model = load_trained_model_from_checkpoint(CONFIG_PATH, CHECKPOINT_PATH, trainable=False)


def build_bert():
    """Builds a hybrid CNN–BiGRU–BERT model for DNA–RNA feature processing."""
    bert_model = load_trained_model_from_checkpoint(CONFIG_PATH, CHECKPOINT_PATH, seq_len=None)

    # Make BERT trainable
    for layer in bert_model.layers:
        layer.trainable = True

    # Inputs
    X_in = Input(shape=(26, 7))
    x1_in = Input(shape=26)
    x2_in = Input(shape=26)

    x_in = Reshape((1, 26, 7))(X_in)
    x_bert = bert_model([x1_in, x2_in])

    # CNN branches
    conv_1 = Conv2D(5, 1, padding="same", activation="relu")(x_in)
    conv_2 = Conv2D(15, 2, padding="same", activation="relu")(x_in)
    conv_3 = Conv2D(25, 3, padding="same", activation="relu")(x_in)
    conv_4 = Conv2D(35, 5, padding="same", activation="relu")(x_in)

    # Merge convolutional outputs
    conv_output = tf.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])
    conv_output = Reshape((26, 80))(conv_output)
    conv_output = Bidirectional(GRU(40, return_sequences=True))(conv_output)

    # Process BERT output
    x_bert = Bidirectional(GRU(40, return_sequences=True))(x_bert)

    # Weighted feature fusion
    feature_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))
    weight_1 = Lambda(lambda x: x * 0.2)
    weight_2 = Lambda(lambda x: x * 0.8)
    x = feature_concat([weight_1(conv_output), weight_2(x_bert)])

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(rate=0.35)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(rate=0.35)(x)

    # Output
    p = Dense(2, activation="softmax")(x)

    # Compile
    model = Model(inputs=[X_in, x1_in, x2_in], outputs=p)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )

    model.summary()
    return model