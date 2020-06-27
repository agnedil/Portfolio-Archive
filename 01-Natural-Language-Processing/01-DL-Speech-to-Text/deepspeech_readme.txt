Source code:

v0.7.3.tar.gz

and the acoustic models:

deepspeech-0.7.3-models.pbmm
deepspeech-0.7.3-models.tflite.

The model with the ".pbmm" extension is memory mapped and thus memory efficient and fast to load. The model with the ".tflite" extension is converted to use TFLite, has post-training quantization enabled, and is more suitable for resource constrained environments.

The acoustic models were trained on American English and the pbmm model achieves an 5.97% word error rate on the LibriSpeech clean test corpus.

In addition we release the scorer:

deepspeech-0.7.3-models.scorer

which takes the place of the language model and trie in older releases.

We also include example audio files:

audio-0.7.3.tar.gz

which can be used to test the engine, and checkpoint files:

deepspeech-0.7.3-checkpoint.tar.gz