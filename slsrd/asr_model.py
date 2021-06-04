import tensorflow as tf

class ASRModel:
    def __init__(self, file_path, device):
        self.graph = tf.compat.v1.Graph()
        self.file_path = file_path
        self.device = device

        self.__load()

    def __load(self):
        tf.compat.v1.reset_default_graph()
        with self.graph.as_default():
            with tf.device(self.device):
                config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.sess = tf.compat.v1.Session(config=config)
                self.saver = tf.compat.v1.train.import_meta_graph(self.file_path+'.meta', clear_devices=True)
                
                self.saver.restore(self.sess, self.file_path)

                self.inputs = tf.compat.v1.get_default_graph().get_tensor_by_name("Placeholder:0")
                self.len_inputs = tf.compat.v1.get_default_graph().get_tensor_by_name("Placeholder_1:0")

                self.extract_layer = tf.compat.v1.get_default_graph().get_tensor_by_name("ForwardPass/w2l_encoder/Relu_9:0")


    def get_asr_feature(self, features):
        with self.graph.as_default():
            latent_feature = self.sess.run(self.extract_layer, 
                feed_dict={self.inputs: [features], 
                           self.len_inputs:[len(features)]}
            )
            return latent_feature