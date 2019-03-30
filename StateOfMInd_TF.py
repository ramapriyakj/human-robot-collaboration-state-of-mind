
import os

import six.moves.urllib.request as request
import tensorflow as tf

# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
print("TensorFlow version: {}".format(tf_version))
assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"

PATH = (os.path.dirname(os.path.realpath(__file__))) + '/files'

#FILE_TRAIN = "/home/amal/robotics/som/somdataset.csv"
tf.logging.set_verbosity(tf.logging.INFO)

feature_names = [
'a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15','a16','a17','a18','a19','a20','a21','a22','a23','a24','a25','a26','a27','a28','a29','a30']

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line,[[0.], [0.], [0.], [0.], [0.],  [0.], [0.], [0.], [0.], [0.],  [0.], [0.], [0.], [0.], [0.],  [0.], [0.], [0.], [0.], [0.],  [0.], [0.], [0.], [0.],  [0.],   [0.], [0.], [0.], [0.], [0.],  [0]])
        label = parsed_line[-1]  # Last element is the label
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything but last elements are the features
        d = dict(zip(feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path)  # Read text file
               .skip(1)  # Skip header row
               .map(decode_csv))  # Transform each elem by applying decode_csv fn
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(32)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

#next_batch = my_input_fn(FILE_TRAIN, True)  # Will return 32 random elements

# Create the feature_columns, which specifies the input to our model
# All our input features are numeric, so use numeric_column for each one
feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

# Create a deep neural network regression classifier
# Use the DNNClassifier pre-made estimator
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,  # The input features to our model
    hidden_units=[30,16,8,4],  # Two layers, each with 10 neurons
    n_classes=4,
    model_dir=PATH)  # Path to where checkpoints etc are stored

# Train our model, use the previously function my_input_fn
# Input to training is a file with training example
# Stop training after 8 iterations of train data (epochs)
#classifier.train(
#    input_fn=lambda: my_input_fn(FILE_TRAIN, True, 8))

prediction_input=[]

def new_input_fn():
    def decode(x):
        x = tf.split(x, 30)  # Need to split into our 4 features
        return dict(zip(feature_names, x))  # To build a dict of them

    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None  # In prediction, we have no labels

def getpred(data):
    global prediction_input
    prediction_input = data
    predict_results = classifier.predict(input_fn=new_input_fn)
    for idx, prediction in enumerate(predict_results):
        if idx<1:
            type = prediction["class_ids"][0]  # Get the predicted class (index)
            break;
    return type;

