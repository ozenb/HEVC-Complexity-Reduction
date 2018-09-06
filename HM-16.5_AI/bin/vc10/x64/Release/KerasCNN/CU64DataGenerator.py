import keras

class CU64DataGenerator(keras.utils.Sequence):
    def __init__(self, path,sampleNumber, batch_size, qp):

        #constants
        self.path = path
        self.batch_size = batch_size
        self.sampleNumber = sampleNumber
        self.cu = 64
        self.cu_2 = 64*64
        self.qp = np.ones((batch_size,1)).dot(qp)
        self.zero_input32 = np.zeros((self.batch_size,32,32,1))
        self.zero_input64 = np.zeros((self.batch_size,64,64,1))
        self.zero_label4 = np.zeros((self.batch_size,4))
        self.zero_label16 = np.zeros((self.batch_size,16))


        #Variables
        self.mean = 0
        self.label1 = np.zeros((self.batch_size,1))
        self.input = np.zeros((self.batch_size,64*64))
        self.down16 = np.zeros((self.batch_size,16*16))

    def __len__(self):
        return int(np.floor(self.sampleNumber / self.batch_size))

    def __getitem__(self,index):
        self.f = open(path,'rb')
        self.down16 = self.down16.reshape((self.batch_size,16*16))

        offset = (1 + self.cu_2)*index
        self.f.seek(offset,0)

        for self.i in range(self.batch_size):
            #Load label
            self.label1[self.i,0] = self.f.read(1)[0]

            #Input
            self.tmp = self.f.read(self.cu_2)

            for self.k in range(self.cu_2):
                #Assign input and decode
                self.input[self.i,self.k] = tmp[self.k]
            #Preprocessing

            #Mean Removal
            self.mean = np.mean(self.input[self.i,:])
            self.input[self.i,:] = self.input[self.i,:] - self.mean

            #Downsampling from 64x64 -> 16x16
            self.down16[self.i,:] = self.input[self.i,::16]
        #End of batching

        #Reshape Input
        self.down16 = self.down16.reshape((self.batch_size,16,16,1))
        self.f.close()

        return [self.down16, self.zero_input32, self.zero_input64, self.qp], [self.label1, self.zero_label4, self.zero_label16]

        
