# Music_genre_classification
Our code is used to take part in the MIREX2017 train/test task.

Our code is based on python3.5.2, tensorflow 1.0.0-rc2. The libraries we used are librosa, numpy, datetime, sys.

This repository contains all my debugging codes and the final version codes.<br>
Our submission record has an extended two code files: extractFeatures.py and TrainAndClassify.py in the mirex_test floder.

The code is all done as required by Submission Format. The code can automatically get the category and the number of categories, so it can be runned in all the sub-tasks.We are based on the genre classification, so you can give priority to the genre sub-tasks.

ExtractFeatures.py receives two command-line arguments, the first is the temporary folder path, and the second is the feature extract file path. <br>
TrainAndClassify.py receives four command line parameters, the first is the temporary folder path, the second is the training set file path, the third is the test set file path, the fourth is the output file path(output file should be created).

The code file run format is as follows:

>>Python extractFeatures.py path1 path2<br>
>>Pyhton TrainAndClassify.py path1 path2 path3 path4

ExtractFeatures code prompt:

>>Preprocessing and Extracting Features ........ All done (means the code was successfully completed.)

TrainAndClassify code Prompt:

>>Training Start (means training starts)<br>
>>Training Done (means training ends)<br>
>>Testing Start (means test starts)<br>
>>Testing Done (means the test ends)<br>

Code will run 100000 iters,run time is about 4 hours with a gpu.
The accurancy can be more than 80% using the GTZAN dataset.
