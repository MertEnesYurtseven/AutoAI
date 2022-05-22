from autokeras import *
Blocks =[ConvBlock(), DenseBlock(), Embedding(), Merge(), ResNetBlock(), RNNBlock(), SpatialReduction(), TemporalReduction(), XceptionBlock(), ImageBlock(), StructuredDataBlock(), TextBlock()]
Nodes=[ImageInput(), Input(), StructuredDataInput(), TextInput()]
Psreprocessor=[ ImageAugmentation(), Normalization(), TextToIntSequence(), TextToNgramVector()]






