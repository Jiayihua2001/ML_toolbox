class simple_data_loader():
    def __init__(self,dataset):
        self.dataset = dataset
        print(f"length of dataset : {len(self.dataset)}")
    def get_batch(self,batch_size):
        self.batch_size = batch_size
        for i in range(0, len(self.dataset), self.batch_size):  #step is batch size
            yield self.dataset[i:i+self.batch_size]


dataset = [i for i in range(1, 21)]
batches = simple_data_loader(dataset).get_batch(7)
for i ,l in enumerate(batches):
    print(f"batch {i} : {l}")
