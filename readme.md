# Deeper LSTM+ normalized CNN for Visual Question Answering

Train a deeper LSTM and normalized CNN Visual Question Answering model. This current code can get **58.16** on Open-Ended and **63.09** on Multiple-Choice on **test-standard** split. You can check [Codalab leaderboard](https://competitions.codalab.org/competitions/6961#results) for more details.

### My own running examples

* preprocess data to raw data :shipit:

```bash
VQA_LSTM_CNN/data$ python vqa_preprocessing_lin.py --split 1
Training sample 248349, Testing sample 121512...
VQA_LSTM_CNN/data$ python vqa_preprocessing_lin.py --split 2
Training sample 369861, Testing sample 244302...
VQA_LSTM_CNN/data$ python vqa_preprocessing_lin.py --split 3
Training sample 369861, Testing sample 60864...
VQA_LSTM_CNN/data$ python vqa_preprocessing_lin_sub.py --split 1
Training sample 36390, Testing sample 18179...
```

* preprocess raw data into question+answer+vocab files :shipit:

```bash
VQA_LSTM_CNN$ python prepro.py --input_train_json data/vqa_raw_s1_train.json --input_test_json data/vqa_raw_s1_test.json --num_ans 1000 --output_json data_prepro_s1_wct0_o1k.json --output_h5 data_prepro_s1_wct0_o1k.h5 --word_count_threshold 0
train question number reduce from 248349 to 215375
total words: 1537357
number of bad words: 0/12603 = 0.00%
number of words in vocab would be 12603
number of UNKs: 0/1537357 = 0.00%

VQA_LSTM_CNN$ python prepro.py --input_train_json data/vqa_raw_s2_train.json --input_test_json data/vqa_raw_s2_test.json --num_ans 1000 --output_json data_prepro_s2_wct0_o1k.json --output_h5 data_prepro_s2_wct0_o1k.h5 --word_count_threshold 0
train question number reduce from 369861 to 320029
total words: 2284620
number of bad words: 0/14770 = 0.00%
number of words in vocab would be 14770
number of UNKs: 0/2284620 = 0.00%

VQA_LSTM_CNN$ python prepro.py --input_train_json data/vqa_raw_s3_train.json --input_test_json data/vqa_raw_s3_test.json --num_ans 1000 --output_json data_prepro_s3_wct0_o1k.json --output_h5 data_prepro_s3_wct0_o1k.h5 --word_count_threshold 0
train question number reduce from 369861 to 320029
total words: 2284620
number of bad words: 0/14770 = 0.00%
number of words in vocab would be 14770
number of UNKs: 0/2284620 = 0.00%

VQA_LSTM_CNN$ python prepro.py --input_train_json data/vqa_raw_s1_subtrain.json --input_test_json data/vqa_raw_s1_subtest.json --num_ans 1000 --output_json data_prepro_s1_wct0_o1k_sub.json --output_h5 data_prepro_s1_wct0_o1k_sub.h5 --word_count_threshold 0
total words: 234395
number of bad words: 0/5444 = 0.00%
number of words in vocab would be 5444
number of UNKs: 0/234395 = 0.00%
```

* generate image features using VGG19/GoogLeNet caffe model :shipit:

```bash
VQA_LSTM_CNN$ th prepro_img_lin.lua -input_json data_prepro_s1_wct0_o1k.json -image_root data/ -cnn_proto /home/deepnet/caffe/models/VGG_19/VGG_ILSVRC_19_layers_deploy.prototxt -cnn_model /home/deepnet/caffe/models/VGG_19/VGG_ILSVRC_19_layers.caffemodel -out_name data_img_s1_VGG19_l43_d4096_o1k.h5 -layer 43 -dim 4096
DataLoader loading h5 file:     data_train
processing 82460 images...
DataLoader loading h5 file:     data_test
processing 40504 images...

VQA_LSTM_CNN$ th prepro_img_lin.lua -input_json data_prepro_s2_wct0_o1k.json -image_root data/ -cnn_proto /home/deepnet/caffe/models/VGG_19/VGG_ILSVRC_19_layers_deploy.prototxt -cnn_model /home/deepnet/caffe/models/VGG_19/VGG_ILSVRC_19_layers.caffemodel -out_name data_img_s2_VGG19_l43_d4096_o1k.h5 -layer 43 -dim 4096
DataLoader loading h5 file:     data_train
processing 122805 images...
DataLoader loading h5 file:     data_test
processing 81434 images...

VQA_LSTM_CNN$ th prepro_img_lin.lua -input_json data_prepro_s3_wct0_o1k.json -image_root data/ -cnn_proto /home/deepnet/caffe/models/VGG_19/VGG_ILSVRC_19_layers_deploy.prototxt -cnn_model /home/deepnet/caffe/models/VGG_19/VGG_ILSVRC_19_layers.caffemodel -out_name data_img_s3_VGG19_l43_d4096_o1k.h5 -layer 43 -dim 4096
DataLoader loading h5 file:     data_train
processing 122805 images...
DataLoader loading h5 file:     data_test
processing 20288 images...

VQA_LSTM_CNN$ th prepro_img_lin_Goo.lua -input_json data_prepro_s1_wct0_o1k.json -split 1 -dim 1000 -input_npy_trainval /home/deepnet/lyt/vqa/feature/VQA/VQA-GoogLeNet-1000.npy -input_npy_test /home/deepnet/lyt/vqa/feature/VQA/VQA-test2015-GoogLeNet-1000.npy -im_path_trainval /home/deepnet/lyt/vqa/feature/VQA/trainval_im_paths.txt -im_path_test /home/deepnet/lyt/vqa/feature/VQA/test_im_paths.txt -out_name data_img_s1_GoogLeNet_d1000_o1k.h5
TrainVal feature size:
 123287x1000
Test feature size:
 123287x1000
actual processing 82460 train image features...
actual processing 40504 test image features...

VQA_LSTM_CNN$ th prepro_img_lin_Goo.lua -input_json data_prepro_s2_wct0_o1k.json -split 2 -dim 1000 -input_npy_trainval /home/deepnet/lyt/vqa/feature/VQA/VQA-GoogLeNet-1000.npy -input_npy_test /home/deepnet/lyt/vqa/feature/VQA/VQA-test2015-GoogLeNet-1000.npy -im_path_trainval /home/deepnet/lyt/vqa/feature/VQA/trainval_im_paths.txt -im_path_test /home/deepnet/lyt/vqa/feature/VQA/test_im_paths.txt -out_name data_img_s2_GoogLeNet_d1000_o1k.h5
TrainVal feature size:
 123287x1000
Test feature size:
 81434x1000
actual processing 122805 train image features...
actual processing 81434 test image features...

VQA_LSTM_CNN$ th prepro_img_lin_Goo.lua -input_json data_prepro_s3_wct0_o1k.json -split 3 -dim 1000 -input_npy_trainval /home/deepnet/lyt/vqa/feature/VQA/VQA-GoogLeNet-1000.npy -input_npy_test /home/deepnet/lyt/vqa/feature/VQA/VQA-test2015-GoogLeNet-1000.npy -im_path_trainval /home/deepnet/lyt/vqa/feature/VQA/trainval_im_paths.txt -im_path_test /home/deepnet/lyt/vqa/feature/VQA/test_im_paths.txt -out_name data_img_s3_GoogLeNet_d1000_o1k.h5
TrainVal feature size:
 123287x1000
Test feature size:
 81434x1000
actual processing 122805 train image features...
actual processing 20288 test image features...

VQA_LSTM_CNN$ th prepro_img_lin_Goo.lua -input_json data_prepro_s1_wct0_o1k.json -split 1 -dim 1024 -input_npy_trainval /home/deepnet/lyt/vqa/feature/VQA/VQA-GoogLeNet-1024.npy -input_npy_test /home/deepnet/lyt/vqa/feature/VQA/VQA-test2015-GoogLeNet-1024.npy -im_path_trainval /home/deepnet/lyt/vqa/feature/VQA/trainval_im_paths.txt -im_path_test /home/deepnet/lyt/vqa/feature/VQA/test_im_paths.txt -out_name data_img_s1_GoogLeNet_d1024_o1k.h5
TrainVal feature size:
 123287x1024
Test feature size:
 123287x1024
actual processing 82460 train image features...
actual processing 40504 test image features...

VQA_LSTM_CNN$ th prepro_img_lin_Goo.lua -input_json data_prepro_s2_wct0_o1k.json -split 2 -dim 1024 -input_npy_trainval /home/deepnet/lyt/vqa/feature/VQA/VQA-GoogLeNet-1024.npy -input_npy_test /home/deepnet/lyt/vqa/feature/VQA/VQA-test2015-GoogLeNet-1024.npy -im_path_trainval /home/deepnet/lyt/vqa/feature/VQA/trainval_im_paths.txt -im_path_test /home/deepnet/lyt/vqa/feature/VQA/test_im_paths.txt -out_name data_img_s2_GoogLeNet_d1024_o1k.h5
TrainVal feature size:
 123287x1024
Test feature size:
 81434x1024
actual processing 122805 train image features...
actual processing 81434 test image features...

VQA_LSTM_CNN$ th prepro_img_lin_Goo.lua -input_json data_prepro_s3_wct0_o1k.json -split 3 -dim 1024 -input_npy_trainval /home/deepnet/lyt/vqa/feature/VQA/VQA-GoogLeNet-1024.npy -input_npy_test /home/deepnet/lyt/vqa/feature/VQA/VQA-test2015-GoogLeNet-1024.npy -im_path_trainval /home/deepnet/lyt/vqa/feature/VQA/trainval_im_paths.txt -im_path_test /home/deepnet/lyt/vqa/feature/VQA/test_im_paths.txt -out_name data_img_s3_GoogLeNet_d1024_o1k.h5
TrainVal feature size:
 123287x1024
Test feature size:
 81434x1024
actual processing 122805 train image features...
actual processing 20288 test image features...
```

* train :shipit:

```bash
VQA_LSTM_CNN$ th train_lin.lua -input_img_h5 data_img_s1_VGG19_l43_d4096_o1k.h5 -input_ques_h5 data_prepro_s1_wct0_o1k.h5 -input_json data_prepro_s1_wct0_o1k.json -max_iters 150000 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -checkpoint_path model/ -CP_name lstm_s1_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k_iter%d.t7 -final_model_name lstm_s1_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k.t7
DataLoader loading h5 file:     data_prepro_s1_wct0_o1k.h5
DataLoader loading h5 file:     data_img_s1_VGG19_l43_d4096_o1k.h5
training loss: 3.4978640579125  on iter: 100/150000
training loss: 3.0087427720755  on iter: 200/150000
...
training loss: 0.61157252263555 on iter: 149900/150000
training loss: 0.61423083797963 on iter: 150000/150000

VQA_LSTM_CNN$ th train_lin.lua -input_img_h5 data_img_s2_VGG19_l43_d4096_o1k.h5 -input_ques_h5 data_prepro_s2_wct0_o1k.h5 -input_json data_prepro_s2_wct0_o1k.json -max_iters 150000 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -checkpoint_path model/ -CP_name lstm_s2_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k_iter%d.t7 -final_model_name lstm_s2_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k.t7
DataLoader loading h5 file:     data_prepro_s2_wct0_o1k.h5
DataLoader loading h5 file:     data_img_s2_VGG19_l43_d4096_o1k.h5
training loss: 3.5151785346515  on iter: 100/150000
training loss: 3.1159727730167  on iter: 200/150000
...
training loss: 0.81721770791108 on iter: 149900/150000
training loss: 0.80389374022118 on iter: 150000/150000

VQA_LSTM_CNN$ th train_lin.lua -input_img_h5 data_img_s3_VGG19_l43_d4096_o1k.h5 -input_ques_h5 data_prepro_s3_wct0_o1k.h5 -input_json data_prepro_s3_wct0_o1k.json -max_iters 150000 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -checkpoint_path model/ -CP_name lstm_s3_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k_iter%d.t7 -final_model_name lstm_s3_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k.t7
DataLoader loading h5 file:     data_prepro_s3_wct0_o1k.h5
DataLoader loading h5 file:     data_img_s3_VGG19_l43_d4096_o1k.h5
training loss: 3.5151785346515  on iter: 100/150000
training loss: 3.1159727730167  on iter: 200/150000
...
training loss: 0.81721770791108 on iter: 149900/150000
training loss: 0.80389374022118 on iter: 150000/150000

VQA_LSTM_CNN$ th train_lin.lua -input_img_h5 data_img_s1_GoogLeNet_l151_d1024_o1k.h5 -imdim 1024 -input_ques_h5 data_prepro_s1_wct0_o1k.h5 -input_json data_prepro_s1_wct0_o1k.json -max_iters 150000 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -checkpoint_path model/ -CP_name lstm_s1_wct0_GoogLeNet_l151_d1024_es200_rs512_rl2_cs1024_o1k_iter%d.t7 -final_model_name lstm_s1_wct0_GoogLeNet_l151_d1024_es200_rs512_rl2_cs1024_o1k.t7
DataLoader loading h5 file:     data_prepro_s1_wct0_o1k.h5
DataLoader loading h5 file:     data_img_s1_GoogLeNet_l151_d1024_o1k.h5
training loss: 3.5614427476161  on iter: 100/150000
training loss: 3.1376313793626  on iter: 200/150000
...
training loss: 0.74972610882572 on iter: 149900/150000
training loss: 0.75323544082434 on iter: 150000/150000

VQA_LSTM_CNN$ th train_lin.lua -input_img_h5 data_img_s2_GoogLeNet_l151_d1024_o1k.h5 -imdim 1024 -input_ques_h5 data_prepro_s2_wct0_o1k.h5 -input_json data_prepro_s2_wct0_o1k.json -max_iters 150000 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -checkpoint_path model/ -CP_name lstm_s2_wct0_GoogLeNet_l151_d1024_es200_rs512_rl2_cs1024_o1k_iter%d.t7 -final_model_name lstm_s2_wct0_GoogLeNet_l151_d1024_es200_rs512_rl2_cs1024_o1k.t7
DataLoader loading h5 file:     data_prepro_s2_wct0_o1k.h5
DataLoader loading h5 file:     data_img_s2_GoogLeNet_l151_d1024_o1k.h5
training loss: 3.6587935287522  on iter: 100/150000
training loss: 3.4186223243636  on iter: 200/150000
...
training loss: 0.93752728053637 on iter: 149900/150000
training loss: 0.93460241173485 on iter: 150000/150000

VQA_LSTM_CNN$ th train_lin.lua -input_img_h5 data_img_s3_GoogLeNet_d1024_o1k.h5 -imdim 1024 -input_ques_h5 data_prepro_s3_wct0_o1k.h5 -input_json data_prepro_s3_wct0_o1k.json -max_iters 150000 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -checkpoint_path model/ -CP_name lstm_s3_wct0_GoogLeNet_d1024_es200_rs512_rl2_cs1024_o1k_iter%d.t7 -final_model_name lstm_s3_wct0_GoogLeNet_d1024_es200_rs512_rl2_cs1024_o1k.t7
DataLoader loading h5 file:     data_prepro_s3_wct0_o1k.h5
DataLoader loading h5 file:     data_img_s3_GoogLeNet_d1024_o1k.h5
training loss: 3.6587935287522  on iter: 100/150000
training loss: 3.4186223243636  on iter: 200/150000
...
training loss: 0.93752728053637 on iter: 149900/150000
training loss: 0.93460241173485 on iter: 150000/150000

VQA_LSTM_CNN$ th train_lin.lua -input_img_h5 data_img_s1_GoogLeNet_d1000_o1k.h5 -imdim 1000 -input_ques_h5 data_prepro_s1_wct0_o1k.h5 -input_json data_prepro_s1_wct0_o1k.json -max_iters 150000 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -checkpoint_path model/ -CP_name lstm_s1_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k_iter%d.t7 -final_model_name lstm_s1_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k.t7
DataLoader loading h5 file:     data_prepro_s1_wct0_o1k.h5
DataLoader loading h5 file:     data_img_s1_GoogLeNet_d1000_o1k.h5
training loss: 3.596404076814   on iter: 100/150000
training loss: 3.1758265550601  on iter: 200/150000
...
training loss: 0.77804907491316 on iter: 149900/150000
training loss: 0.77817997602014 on iter: 150000/150000

VQA_LSTM_CNN$ th train_lin.lua -input_img_h5 data_img_s2_GoogLeNet_d1000_o1k.h5 -imdim 1000 -input_ques_h5 data_prepro_s2_wct0_o1k.h5 -input_json data_prepro_s2_wct0_o1k.json -max_iters 150000 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -checkpoint_path model/ -CP_name lstm_s2_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k_iter%d.t7 -final_model_name lstm_s2_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k.t7
DataLoader loading h5 file:     data_prepro_s2_wct0_o1k.h5
DataLoader loading h5 file:     data_img_s2_GoogLeNet_d1000_o1k.h5
training loss: 3.5959485306648  on iter: 100/150000
training loss: 3.281319634216   on iter: 200/150000
...
training loss: 0.93689886374299 on iter: 149900/150000
training loss: 0.9203415711874  on iter: 150000/150000

VQA_LSTM_CNN$ th train_lin.lua -input_img_h5 data_img_s3_GoogLeNet_d1000_o1k.h5 -imdim 1000 -input_ques_h5 data_prepro_s3_wct0_o1k.h5 -input_json data_prepro_s3_wct0_o1k.json -max_iters 150000 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -checkpoint_path model/ -CP_name lstm_s3_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k_iter%d.t7 -final_model_name lstm_s3_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k.t7
DataLoader loading h5 file:     data_prepro_s3_wct0_o1k.h5
DataLoader loading h5 file:     data_img_s3_GoogLeNet_d1000_o1k.h5
training loss: 3.5959485306648  on iter: 100/150000
training loss: 3.281319634216   on iter: 200/150000
...
training loss: 0.93689886374299 on iter: 149900/150000
training loss: 0.9203415711874  on iter: 150000/150000
```

* evaluation :shipit:

```bash
VQA_LSTM_CNN$ th eval_lin.lua -input_img_h5 data_img_s1_VGG19_l43_d4096_o1k.h5 -input_ques_h5 data_prepro_s1_wct0_o1k.h5 -input_json data_prepro_s1_wct0_o1k.json -model_path model/lstm_s1_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k.t7 -imdim 4096 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -result_name lstm_s1_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k_results.json
 [====================================== 121512/121512 ===========================>]  Tot: 16s759ms | Step: 0ms
save results in: result/OpenEnded_lstm_s1_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k_results.json
save results in: result/MultipleChoice_lstm_s1_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k_results.json

VQA_LSTM_CNN$ th eval_lin.lua -input_img_h5 data_img_s2_VGG19_l43_d4096_o1k.h5 -input_ques_h5 data_prepro_s2_wct0_o1k.h5 -input_json data_prepro_s2_wct0_o1k.json -model_path model/lstm_s2_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k.t7 -imdim 4096 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -result_name lstm_s2_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k_results.json
 [====================================== 244302/244302 ===========================>]  Tot: 36s64ms | Step: 0ms
save results in: result/OpenEnded_lstm_s2_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k_results.json
save results in: result/MultipleChoice_lstm_s2_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k_results.json

VQA_LSTM_CNN$ th eval_lin.lua -input_img_h5 data_img_s3_VGG19_l43_d4096_o1k.h5 -input_ques_h5 data_prepro_s3_wct0_o1k.h5 -input_json data_prepro_s3_wct0_o1k.json -model_path model/lstm_s3_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k.t7 -imdim 4096 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -result_name lstm_s3_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k_results.json
 [====================================== 60864/60864 ===========================>]  Tot: 9s754ms | Step: 0ms
save results in: result/OpenEnded_lstm_s3_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k_results.json
save results in: result/MultipleChoice_lstm_s3_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k_results.json

VQA_LSTM_CNN$ th eval_lin.lua -input_img_h5 data_img_s1_GoogLeNet_l151_d1024_o1k.h5 -imdim 1024 -input_ques_h5 data_prepro_s1_wct0_o1k.h5 -input_json data_prepro_s1_wct0_o1k.json -model_path model/lstm_s1_wct0_GoogLeNet_l151_d1024_es200_rs512_rl2_cs1024_o1k.t7 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -result_name lstm_s1_wct0_GoogLeNet_l151_d1024_es200_rs512_rl2_cs1024_o1k_results.json
 [====================================== 121512/121512 ===========================>]  Tot: 15s935ms | Step: 0ms
save results in: result/OpenEnded_lstm_s1_wct0_GoogLeNet_l151_d1024_es200_rs512_rl2_cs1024_o1k_results.json
save results in: result/MultipleChoice_lstm_s1_wct0_GoogLeNet_l151_d1024_es200_rs512_rl2_cs1024_o1k_results.json

VQA_LSTM_CNN$ th eval_lin.lua -input_img_h5 data_img_s2_GoogLeNet_l151_d1024_o1k.h5 -imdim 1024 -input_ques_h5 data_prepro_s2_wct0_o1k.h5 -input_json data_prepro_s2_wct0_o1k.json -model_path model/lstm_s2_wct0_GoogLeNet_l151_d1024_es200_rs512_rl2_cs1024_o1k.t7 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -result_name lstm_s2_wct0_GoogLeNet_l151_d1024_es200_rs512_rl2_cs1024_o1k_results.json
 [====================================== 244302/244302 ===========================>]  Tot: 34s545ms | Step: 0ms
save results in: result/OpenEnded_lstm_s2_wct0_GoogLeNet_l151_d1024_es200_rs512_rl2_cs1024_o1k_results.json
save results in: result/MultipleChoice_lstm_s2_wct0_GoogLeNet_l151_d1024_es200_rs512_rl2_cs1024_o1k_results.json

VQA_LSTM_CNN$ th eval_lin.lua -input_img_h5 data_img_s3_GoogLeNet_d1024_o1k.h5 -imdim 1024 -input_ques_h5 data_prepro_s3_wct0_o1k.h5 -input_json data_prepro_s3_wct0_o1k.json -model_path model/lstm_s3_wct0_GoogLeNet_d1024_es200_rs512_rl2_cs1024_o1k.t7 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -result_name lstm_s3_wct0_GoogLeNet_d1024_es200_rs512_rl2_cs1024_o1k_results.json
 [====================================== 60864/60864 ===========================>]  Tot: 8s968ms | Step: 0ms
save results in: result/OpenEnded_lstm_s3_wct0_GoogLeNet_d1024_es200_rs512_rl2_cs1024_o1k_results.json
save results in: result/MultipleChoice_lstm_s3_wct0_GoogLeNet_d1024_es200_rs512_rl2_cs1024_o1k_results.json

VQA_LSTM_CNN$ th eval_lin.lua -input_img_h5 data_img_s1_GoogLeNet_d1000_o1k.h5 -imdim 1000 -input_ques_h5 data_prepro_s1_wct0_o1k.h5 -input_json data_prepro_s1_wct0_o1k.json -model_path model/lstm_s1_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k.t7 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -result_name lstm_s1_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k_results.json
 [====================================== 121512/121512 ===========================>]  Tot: 16s76ms | Step: 0ms
save results in: result/OpenEnded_lstm_s1_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k_results.json
save results in: result/MultipleChoice_lstm_s1_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k_results.json

VQA_LSTM_CNN$ th eval_lin.lua -input_img_h5 data_img_s2_GoogLeNet_d1000_o1k.h5 -imdim 1000 -input_ques_h5 data_prepro_s2_wct0_o1k.h5 -input_json data_prepro_s2_wct0_o1k.json -model_path model/lstm_s2_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k.t7 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -result_name lstm_s2_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k_results.json
 [====================================== 244302/244302 ===========================>]  Tot: 36s524ms | Step: 0ms
save results in: result/OpenEnded_lstm_s2_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k_results.json
save results in: result/MultipleChoice_lstm_s2_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k_results.json

VQA_LSTM_CNN$ th eval_lin.lua -input_img_h5 data_img_s3_GoogLeNet_d1000_o1k.h5 -imdim 1000 -input_ques_h5 data_prepro_s3_wct0_o1k.h5 -input_json data_prepro_s3_wct0_o1k.json -model_path model/lstm_s3_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k.t7 -input_encoding_size 200 -rnn_size 512 -rnn_layer 2 -common_embedding_size 1024 -num_output 1000 -result_name lstm_s3_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k_results.json
 [====================================== 60864/60864 ===========================>]  Tot: 8s913ms | Step: 0ms
save results in: result/OpenEnded_lstm_s3_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k_results.json
save results in: result/MultipleChoice_lstm_s3_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k_results.json
```

* Score :shipit:

```bash
VQA/PythonEvaluationTools$ python vqaEvalDemo_lin.py --taskType MultipleChoice --dataType mscoco --dataSubType val2014 --intermediateType s1_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k --resultType lstm
Overall Accuracy is: 59.30
Per Answer Type Accuracy is the following:
other : 50.19
number : 34.24
yes/no : 79.88

VQA/PythonEvaluationTools$ python vqaEvalDemo_lin.py --taskType MultipleChoice --dataType mscoco --dataSubType val2014 --intermediateType s1_wct0_GoogLeNet_l151_d1024_es200_rs512_rl2_cs1024_o1k --resultType lstm
Overall Accuracy is: 59.31
Per Answer Type Accuracy is the following:
other : 50.27
number : 34.51
yes/no : 79.71

VQA/PythonEvaluationTools$ python vqaEvalDemo_lin.py --taskType MultipleChoice --dataType mscoco --dataSubType val2014 --intermediateType s1_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k --resultType lstm
Overall Accuracy is: 59.01
Per Answer Type Accuracy is the following:
other : 49.67
number : 34.15
yes/no : 79.84

VQA/PythonEvaluationTools$ python vqaEvalDemo_lin.py --taskType MultipleChoice --dataType mscoco --dataSubType subval2014 --intermediateType s1_wct0_VGG19_l43_d4096_es200_rs512_rl2_cs1024_o1k_sub --resultType lstm
Overall Accuracy is: 51.72
Per Question Type Accuracy is the following:
what : 38.63
is this : 70.29
is there a : 84.33
what is the : 41.48
is the : 71.07
is this a : 74.04
what kind of : 44.03
what is : 36.58
are the : 68.72
how many : 35.34
what color is the : 44.40
Per Answer Type Accuracy is the following:
other : 42.19
number : 33.24
yes/no : 75.33

VQA/PythonEvaluationTools$ python vqaEvalDemo_lin.py --taskType MultipleChoice --dataType mscoco --dataSubType subval2014 --intermediateType s1_wct0_GoogLeNet_d1024_es200_rs512_rl2_cs1024_o1k_sub --resultType lstm
Overall Accuracy is: 52.24
Per Question Type Accuracy is the following:
what : 39.50
is this : 70.48
is there a : 85.14
what is the : 40.81
is the : 70.51
is this a : 74.15
what kind of : 46.42
what is : 40.24
are the : 68.88
how many : 35.43
what color is the : 45.68
Per Answer Type Accuracy is the following:
other : 43.01
number : 33.54
yes/no : 75.56

VQA/PythonEvaluationTools$ python vqaEvalDemo_lin.py --taskType MultipleChoice --dataType mscoco --dataSubType subval2014 --intermediateType s1_wct0_GoogLeNet_d1000_es200_rs512_rl2_cs1024_o1k_sub --resultType lstm
Overall Accuracy is: 51.55
Per Question Type Accuracy is the following:
what : 37.98
is this : 70.40
is there a : 85.87
what is the : 40.36
is the : 70.20
is this a : 73.49
what kind of : 45.37
what is : 39.04
are the : 69.87
how many : 34.71
what color is the : 44.22
Per Answer Type Accuracy is the following:
other : 42.02
number : 32.88
yes/no : 75.28
```

### Requirements

This code is written in Lua and requires [Torch](http://torch.ch/). The preprocssinng code is in Python, and you need to install [NLTK](http://www.nltk.org/) if you want to use NLTK to tokenize the question.

### Ubuntu installation

Installation on other platforms is likely similar, although the command won't be `apt-get`, and the names of the system packages may be different.

```bash
sudo apt-get install libpng-dev libtiffio-dev libhdf5-dev
pip install pillow
pip install -r requirements.txt
python -c "import nltk; nltk.download('all')"
```

### Troubleshooting the prepro.py installation

If when running `prepro.py` you get the error:

```AttributeError: 'module' object has no attribute 'imread'```

Then this means that you're missing the Python library `pillow`. Confusingly, `scipy` rewrites its `scipy.misc` module
depending on what modules are available when the library was installed. To fix it, do:

```bash
pip uninstall scipy
pip install pillow
pip install --no-cache-dir scipy
```

### Evaluation

We have prepared everything for you. 

If you want to train on **train set** and evaluate on **validation set**, you can download the features from [here](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_only/data_train_val.zip), and the pretrained model from [here](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_only/pretrained_lstm_train_val.t7.zip), put it under the main folder, run 

```
$ th eval.lua -input_img_h5 data_img.h5 -input_ques_h5 data_prepro.h5 -input_json data_prepro.json -model_path model/lstm.t7
```

This will generate the answer json file both on Open-Ended and Multiple-Choice. To evaluate the accuracy of generate result, you need to download the [VQA evaluation tools](https://github.com/VT-vision-lab/VQA).

If you want to train on **train + validation set** and evaluate on **test set**, you can download the feature from [here](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_val/data_train-val_test.zip), and the pretrained model from [here](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_val/pretrained_lstm_train-val_test). The rest is the same as previous one, except you need to evaluate the generated results on [testing server](http://www.visualqa.org/challenge.html).

### Training

The first thing you need to do is to download the data and do some preprocessing. Head over to the `data/` folder and run 

```
$ python vqa_preprocessing.py --download True --split 1
```

`--download Ture` means you choose to download the VQA data from the [VQA website](http://www.visualqa.org/) and `--split 1` means you use COCO train set to train and validation set to evaluation. `--split 2 ` means you use COCO train+val set to train and test set to evaluate. After this step, it will generate two files under the `data` folder. `vqa_raw_train.json` and `vqa_raw_test.json`

Once you have these, we are ready to get the question and image features. Back to the main folder, run

```
$ python prepro.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --num_ans 1000
``` 

to get the question features. `--num_ans` specifiy how many top answers you want to use during training. You will also see some question and answer statistics in the terminal output. This will generate two files in your main folder, `data_prepro.h5` and `data_prepro.json`. To get the image features, run

```
$ th prepro_img.lua -input_json data_prepro.json -image_root path_to_image_root -cnn_proto path_to_cnn_prototxt -cnn_model path to cnn_model
```

Here we use VGG_ILSVRC_19_layers [model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77). After this step, you can get the image feature `data_img.h5`. We have prepared everything and ready to launch training. You can simply run

```
$ th train.lua
``` 

with the default parameter, this will take several hours on a sinlge Tesla k40 GPU, and will generate the model under `model/save/`

### Reference

If you use this code as part of any published research, please acknowledge the following repo
```
@misc{Lu2015,
author = {Jiasen Lu and Xiao Lin and Dhruv Batra and Devi Parikh},
title = {Deeper LSTM and normalized CNN Visual Question Answering model},
year = {2015},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/VT-vision-lab/VQA_LSTM_CNN}},
commit = {6c91cb9}
}
```
If you use the VQA dataset as part of any published research, please acknowledge the following paper
```
@InProceedings{Antol_2015_ICCV,
author = {Stanislaw Antol and Aishwarya Agrawal and Jiasen Lu and Margaret Mitchell and Dhruv Batra and C. Lawrence Zitnick and Devi Parikh},
title = {VQA: Visual Question Answering},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {December},
year = {2015}
}
```
### License

BSD License.
