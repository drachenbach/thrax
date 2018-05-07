# Thrax - A Framework for Link Prediction in Knowledge Graphs with Embedding Models

## How To Build
```
$ mkdir <build-dir> && cd <build-dir>
$ cmake ..
$ make
$ ./main -h
```
Create a build directory, use `cmake` to produce a `Makefile` from `CMakeLists.txt`. Use `make` to build project and then execute the binary.

## Program Flow
The design of the framework is modular, meaning that each module is meant to be exchangeable by exposing a consistent API.

Each of the modules can be specified and configured in the `config.json` file. The first thing is reading and parsing this file.

### Load Data
The triple data (only true triples) is assumed to be stored in a single directory `data.dir`. In this directory there should be three files: train, validation, and test data. Each of the files must be tab separated and must contain a single triple per line.

The string representations in the files will be mapped to unique integer IDs to work with inside the framework. It is also possible to specify these mappings `data.loadMappings` by creating a two files in a directory: one mapping entity names to IDs named `entityMappings.csv`, the other mapping relation names to IDs named `relationMappings.csv`. The IDs have to start at 0 and count up till the number of relations - 1, as they are used for array indexing. Mappings will optionally be dumped to file as well `data.dumpMappings`.

There is also an option `optimizer.trainOnValidation` which will append the validation to the train data to train on the combined dataset. Note: do not use this in combination with early stopping.

There is also an option `optimizer.testOnValidation` which performs the evaluation on the validation data. This is especially helpful for grid search for hyperparameter tuning.

### Model
In the next step, the `ModelFactory` will build a model that is specified by the `model` entry in the `config.json` file.

The model's parameters can be (optionally) initialized by pre-trained parameters that were already dumped to disk by using `model.initializer.type = 'pretrained'` and specifying the `model.initializer.location`.

### Optional: Gradient Checking
The `checkGradients` option will trigger to validate the model's implementation. It will verify that the implemented gradient matches the scoring function by empirically approximating the gradient.

### Optimizer
If a model should be trained the optimizer will be required. It takes train and validation data, the model and the config and will perform gradient descent.

The optimizer will start the training loop for a maximum of `optimizer.maxEpochs` epochs. In each epochs it shuffles the data and splits up the (true) triples into minibatches of size `optimizer.batchSize`.

For each true triple in the mini-batch the sampler will be used to sample `optimizer.sampling.numberOfNegatives` negative triples according to the specified sampling strategy `optimizer.sampling.type`. The `optimizer.sampling.type` can be used to configure which constituent will be perturbed: `subject` always perturbs only the subject, `object` always perturbs only the object, `random` randomly selects between subject and object, `both` always perturbs the subject *and* object. 

After that, the gradients of the specified loss function `optimizer.type` w.r.t. the model parameters are calculated using the generated batch of positive and negative triples.

With the obtained gradients, in the next step the model parameters have to be updated. To calculate this update a parameter updater `model.update` is used. It can be vanilla SGD or adaptive learning rate algorithms such as AdaGrad, AdaDelta, or RMSProp.

Model parameters are updated.

#### Early Stopping
Optionally, the optimizer is performing early stopping `optimizer.earlyStopping.useEarlyStopping`. It will estimate the model's current performance every `optimizer.earlyStopping.everyNEpochs` epochs on `optimizer.earlyStopping.n` triples of the validation data by calculating the raw mean reciprocal rank. If the performance does not increase, training will be stopped.

#### Hooks
The optimizer provides two hooks that can be implemented by the models:
1. Post batch: this hook is called after the parameter update of one batch.
2. Post epoch: this hook is called after each epoch.

#### Serialization
The framework always dumps the learned models after training finished (or the best model when using early stopping). This includes the used config file, detailed evaluation metrics, the embedding parameters, as well as statistics about the training (training time, loss and early stopping metrics per epoch). The location can be configured by `serialization.dumpDirectory` and `serialization.dumpLocation`. `serialization.dumpDirectory` can be seen as the root of the output which defaults to `auto` (dumps to `./models`) but can be adapted which is especially helpful for grid search. `serialization.dumpLocation` is a directory that is created in the `serialization.dumpDirectory`, it is recommended to use `auto` which will generate a unique model name.

### Evaluation
In the last step, evaluation is performed. Therefore, the test data is loaded.

#### Calculate Ranks
For evaluation, each test triple is analyzed separately. First the object is replaced by all possible entities, for each such triple the model's score is calculated. These scores are sorted and the rank of the true triple is stored. The same procedure is done by replacing the subject.

#### Calculate Metrics
The obtained ranks are grouped by several criteria and metrics are calculated on each of the groups. Groups:
* Predicting object: all ranks that are obtained when replacing the object.
* Predicting subject: all ranks that are obtained when replacing the subject.
* Predicting both: concatenates the ranks of the previous two and then calculates metrics.
* All of the above three grouped by relation.

Each group is again split into two according to a settings:
1. Raw: calculate the metrics for all possible triples.
2. Filtered: calculate the metrics only on triples that *do not* occur in train, validation, or test data (target triple excluded).

In the end, there are 6 global groups and 6 groups per relation. On each of the groups the following metrics will be calculated:
* Mean rank
* Mean reciprocal rank
* Hits@1
* Hits@10

All of the metrics will be dumped to disk.

## Implementing Your Own Model
The framework assumes each model to provide a scoring function that assigns a (real-valued) score to a triple. Higher scores mean that a triple's existence is more likely.

Within this assumption, implementing your own model is easy. You essentially have to specify the model's scoring function and its gradient.

The best way to get started is having a look at the already existing implementation of models.

### Hyper Parameters
The model has access to the complete config. Hyper parameters need to be extracted in the model's constructor e.g.
```c++
int k = config.get<int>("model.hyperParameters.k");
```

### Register Parameters
The framework assumes that each model can consist of several sets of embedding parameters e.g. entity embeddings. These parameters need to be known by the framework, thus you have to register them e.g. in the constructor
```c++
addParameter("E", k, n, ENTITY);
```
This adds a parameter set named `E` with embedding dimension `k` and size `n`. It is tagged to be associated with an `ENTITY`. Internally, each parameter set is stored as a `kxn` dense [Eigen](http://eigen.tuxfamily.org/) matrix where each column corresponds to the embedding of one constituent.

Note that you can add multiple parameter sets (also multiple ones for e.g. entities).

#### Parameter Access
You can access the parameter sets easily. I recommend holding a pointer to each parameter set in the model's class.
```c++
EmbeddingParameterSet* E = &(getParameter("E"));
```
Then you can access the embedding of one constituent (column of a parameter set) e.g. like this
```c++
E->col(triple.subject);
```

###### Matrix-valued Embeddings
As all embeddings are essentially stored as one column of a parameter set matrix, it makes it easy for vector-valued embeddings. However, some models (e.g. RESCAL) have matrix-valued embeddings which require casting using [Eigen Maps](http://eigen.tuxfamily.org/dox/group__TutorialMapClass.html) e.g. 
```c++
Map<MatrixXd>(R->col(triple.relation).data(), k, k)
```
will select the embedding of the triple's relation and interpret it as a `kxk` matrix.

### Scoring Function
The `score` functions takes a triple and returns the `double` score that the model assigns to it.

#### Performance Optimization
Some models allow to optimize the evaluation by computing the scores of multiple triples more efficiently. Therefore, a model can override `scoreSubjectRelation` and `scoreRelationObject`. See e.g. in the RESCAL implementation.

### Gradient Function
The `gradient` function takes a triple and a `scale`, calculates the appropriate gradients and stores them.

Each of the parameter sets has a corresponding gradient data structure with the same name. I also recommend here, to hold a pointer in the model's class:
```c++
Gradient* dE = &(getGradient("E"));
```
These data structures are used to store the gradients of the embedding parameters of a single constituent.

You should calculate the gradients for subject, relation, and object embeddings e.g. in DISTMULT the subject embedding gradient of the scoring function is 
```c++
VectorXd subjectGradient = R->col(triple.relation).array() * E->col(triple.object).array();
```
So simply the element-wise product of the relation and object embedding. This gradient needs to stored by calling
```c++
dE->add(triple.subject, scale*subjectGradient);
```
This tells the gradient data structure of the entity embeddings to add `scale*subjectGradient` for the triple's subject entity.

**Important**: you *have to* multiply the your calculated gradient of the scoring function by the `scale` scalar as it is used by the loss function to calculate the overall gradient.

### Regularization
To perform regularization, two approaches are common:
1. Normalizing embeddings to have unit norm.
2. Use L2 regularization.

L2 regularization is handled by the framework and assumes that there are `model.hyperParameters.lambda_e` and `model.hyperParameters.lambda_e` are configured. Both default to `0.0`.
Normalizing the embeddings is handled by the framework and assumes that `model.hyperParameters.normalizeEntities` and `model.hyperParameter.normalizeRelations` are configured. Both default to `false`.

## Ensembles
It is also possible to train ensembles of individual embeddings models. Therefore, `model.type` has to be `ensemble`. In `model.models` the individual models can be configured (have a look at the example `config/config-ensemble.json`).

The `model.trainModels` specifies whether the individual models should be trained (fine-tuned) as well. If not, only the ensemble parameters (such as weights) are trained.

An ensemble itself can, of course, have hyper parameters `model.hyperParameters`.

The `model.update` sub-tree will be pasted in each model in `model.models` for a consistent training procedure.

## Grid Search
The Jupyter notebook `GridSearchAndResultProcessing.ipynb` gives examples of how to use Python to perform grid search on multiple cores. Also, it contains examples of how to process the produced results, including plotting examples.

## Repository Structure
```
│   README.md
│   CMakeLists.txt
|   GridSearchAndResultProcessing.ipynb: Python Jupyter otebook for grid search and result processing
|
└───extern/cmake: Files to help CMake to find MKL and TBB
│
└───config
│   │   config.json: annotated example config file
|   |   other example config files
│
└───src
│   │   main.cpp: main program that triggers all computations
│
└───include/thrax: framework implementation
│   │
│   └───evaluation: perform model evaluation
│   │
│   └───intializer: parameter initialization module
│   │
│   └───lossFunction: several loss function implementations
│   │
│   └───model: base model implementation and specific model implementations
│   │
│   └───optimizer: gradient descent optimizer implementation
│   │
│   └───parameterUpdater: implementations of variants of gradient descent, also with adaptive learning rates
│   │
│   └───sampler: sampling module, implements how negative triples are sampled
│   │
│   └───struct: helpful data structures
│   │
│   └───util: helpful utility functions
```
