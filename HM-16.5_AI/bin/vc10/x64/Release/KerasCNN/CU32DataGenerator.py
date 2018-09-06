class CU32DataGenerator(keras.utils.Sequence):
    def __init__(self, path, sampleNumber, batch_size,qp):

        #constants
        self.path = path
        self.batch_size = batch_size
        self.sampleNumber = sampleNumber
        self.cu = 32
        self.cu_2 = 32*32
        self.qp = np.ones((self.batch_size,1)).dot(qp)

        self.zero_input16 = np.zeros((self.batch_size,16,16,1))
        self.zero_input64 = np.zeros((self.batch_size,64,64,1))
        self.zero_label1 = np.zeros((self.batch_size,1))
        self.zero_label16 = np.zeros((self.batch_size,16))

        #Variables
        self.mean = 0
        self.label4 = np.zeros((self.batch_size,4))
        self.input = np.zeros((self.batch_size,))
