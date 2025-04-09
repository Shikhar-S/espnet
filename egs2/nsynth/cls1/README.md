# Nsynth Dataset for Instrument & Pitch Classification Recipe

This recipe implements the audio classification task with a BEATs encoder and linear layer decoder model on the [Nsynth](https://magenta.tensorflow.org/datasets/nsynth) dataset. 

	@misc{nsynth2017,
    Author = {Jesse Engel and Cinjon Resnick and Adam Roberts and
              Sander Dieleman and Douglas Eck and Karen Simonyan and
              Mohammad Norouzi},
    Title = {Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders},
    Year = {2017},
    Eprint = {arXiv:1704.01279},
}

We reuse part of the code from the [BEATs repository](https://github.com/microsoft/unilm/tree/master/beats) for this implementation.

# Trained checkpoints
Fine-tuned checkpoint is available at:
* https://huggingface.co/espnet/BEATs-AS20K 


