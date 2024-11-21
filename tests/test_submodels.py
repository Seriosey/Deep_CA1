import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.src.backend import shape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, RNN, Layer
from tensorflow.keras.saving import load_model


import os
os.chdir("../")
from synapses_layers import TsodycsMarkramSynapse

class TimeStepLayer(Layer):

    def __init__(self, units, base_pop_model, synapses_params, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

        self.base_pop_model = base_pop_model
        self.synapses_params = synapses_params


    def get_initial_state(self, batch_size=1):
        state = K.zeros(shape=(batch_size, self.state_size), dtype=tf.float32)
        state = tf.convert_to_tensor(state)
        return state

    def get_pop_model_with_synapses(self, input_shape, syn_params):

        dt = 0.1

        ndimsinp = syn_params["gsyn_max"].size

        mask = np.ones(ndimsinp, dtype=bool)

        synapses = TsodycsMarkramSynapse(syn_params, dt=dt, mask=mask)
        synapses_layer = RNN(synapses, return_sequences=True, stateful=True, name="Synapses_Layer")

        input_layer = Input(shape=(None, ndimsinp), batch_size=1)
        synapses_layer = synapses_layer(input_layer)

        base_model = tf.keras.models.clone_model(self.base_pop_model)

        model = Model(inputs=input_layer, outputs=base_model(synapses_layer), name="Population_with_synapses")

        return model



    def build(self, input_shape):

        self.batch_size = input_shape[0]
        self.n_dims = input_shape[-1]

        self.pop_models = []
        for syn_params in self.synapses_params:


            model = self.get_pop_model_with_synapses(input_shape, syn_params)
            # model = Sequential()
            # model.add(Input(shape=(None, input_shape[-1]+self.state_size)))
            # model.add(GRU(16, return_sequences=True, stateful=False) )  #
            # model.add(GRU(16, return_sequences=True, stateful=False))  #
            # model.add(Dense(1, activation='relu'))  #
            self.pop_models.append(model)

    def call(self, input, state):


        input = K.concatenate([state[0], input], axis=-1)
        input = K.reshape(input, shape=(1, 1, -1))

        output = []
        for model in self.pop_models:
            out = model(input)
            output.append(out)
        output = K.concatenate(output, axis=-1)

        return output, output[0]


# # Параметры данных
Ns = 5
ext_input = 3

full_size = Ns+ext_input

params = {
    "gsyn_max": np.zeros(full_size, dtype=np.float32) + 1.5,
    "Uinc": np.zeros(full_size, dtype=np.float32) + 0.5,
    "tau_r": np.zeros(full_size, dtype=np.float32) + 1.5,
    "tau_f": np.zeros(full_size, dtype=np.float32) + 1.5,
    "tau_d": np.zeros(full_size, dtype=np.float32) + 1.5,
    'pconn': np.zeros(full_size, dtype=np.float32) + 1.0,
    'Erev': np.zeros(full_size, dtype=np.float32),
    'Cm': 0.114,
    'Erev_min': -75.0,
    'Erev_max': 0.0,
}

synapse_params = [params for _ in range(Ns)]

base_model = load_model("./pretrained_models/NO_Trained.keras")
for layer in base_model.layers:
    layer.trainable = False

time_step_layer = TimeStepLayer(Ns, base_model, synapse_params)


big_model = Sequential()
my_layer = RNN(time_step_layer, return_sequences=True, stateful=True)

big_model.add(Input(shape=(None, ext_input), batch_size=1))
big_model.add(my_layer)
big_model.compile(loss="mean_squared_logarithmic_error", optimizer="adam")

#print(big_model.trainable_variables)


timesteps = 100
# Генерация случайных входных данных
X = np.random.rand(1, timesteps, ext_input)


# Генерация ответов
Y = np.random.rand(timesteps, Ns).reshape(1, timesteps, Ns)

X = tf.convert_to_tensor(value=X, dtype='float32')
y_pred = big_model.predict(X)
#hist = big_model.fit(X, Y, epochs=2)

print(y_pred.shape)



