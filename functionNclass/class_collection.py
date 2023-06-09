class VideoDatasetSourceAndTarget:
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.output_dim = (source_dataset.output_dim, target_dataset.output_dim)
        # Get the size of the tensors for the first sample in the source and target datasets
        source_size = len(source_dataset[0][0]), *source_dataset[0][0].size()[1:]
        target_size = len(target_dataset[0][0]), *target_dataset[0][0].size()[1:]

        # Store the size of the tensors as instance variables
        self.source_size = source_size
        self.target_size = target_size

    def __len__(self):
        return max([len(self.source_dataset), len(self.target_dataset)])

    def __getitem__(self, index):
        source_index = index % len(self.source_dataset)
        source_data = self.source_dataset[source_index]

        target_index = index % len(self.target_dataset)
        target_data = self.target_dataset[target_index]
        return (source_index, *source_data, target_index, *target_data)

