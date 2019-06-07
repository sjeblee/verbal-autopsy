# verbal-autopsy

<h1>Verbal Autopsy pipeline</h1>
Developed by Serena Jeblee (sjeblee@cs.toronto.edu) at University of Toronto

<h2>NOTE: We are currently in the process of upgrading the code to python 3.6.</h2>

<h3>Run the pipeline:</h3>
./pipeline.sh

pipeline.py: main pipeline script

spellcorrect.py: spelling correction for narratives and keywords

extract_features.py: feature extraction

model.py: passes the data to the model and set parameters

model_library.py: the actual model functions (Keras w/ Theano backend)

model_library_torch.py: model functions in PyTorch

results_stats.py: calculates metrics from predicted labels

<h3>Requires:</h3>

Python 2.7 and the following Python modules:
keras
nltk
numpy
scikit-learn
theano

For PyTorch models (model_library_torch), you will need python 3 and:
pytorch
numpy

<h3>Other module-specific requirements:</h3>

spellcorrect: requires pyenchant

hyperopt function in model requires hyperopt

rebalance option requires imbalanced-learn

Use of the Heideltime tagger and Stanford Dependency parser also requires Java
