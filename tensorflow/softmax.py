import tensorflow as tf

class DifferentiatedSoftmaxModel(tf.keras.Model):
    def __init__(self, num_class, data_dim=32, hidden_dim=64, hidden_portions=[0.5, 0.35, 0], class_portions=[0.2, 0.3, 0]):
        super(DifferentiatedSoftmaxModel, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim)
        self.final_hiddens = [int(hidden_dim * portion) if portion != 0 else 0 for portion in hidden_portions]
        self.final_hiddens[-1] = hidden_dim - sum(self.final_hiddens[:-1])
        self.class_bins = [int(num_class * portion) if portion != 0 else 0 for portion in class_portions]
        self.class_bins[-1] = num_class - sum(self.class_bins[:-1])
        self.fc_finals = [tf.keras.layers.Dense(class_bin) for class_bin in self.class_bins]

    def call(self, x, return_logits=False):
        x = self.fc1(x)
        x_splitted = []
        t = 0
        for final_hidden in self.final_hiddens:
            x_splitted.append(x[:, t:t + final_hidden])
            t += final_hidden
        
        x = tf.concat([fc_final(x_split) for x_split, fc_final in zip(x_splitted, self.fc_finals)], axis=-1)

        if return_logits:
            return x
        return tf.nn.softmax(x, axis=-1)
