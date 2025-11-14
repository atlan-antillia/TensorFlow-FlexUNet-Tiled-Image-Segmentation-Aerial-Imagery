<h2>TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery (2025/11/14)</h2>

Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Image Segmentation for <b>Aerial Imagery Aerial-Imagery</b> (Singleclass)  based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 512x512  pixels 
<a href="https://drive.google.com/file/d/1cZ9-JmCyJ8jjmErQ2ijvo5y4JT6AZ-TW/view?usp=sharing">
<b>Augmented-Tiled-Aerial-Imagery-ImageMask-Dataset.zip</b></a>
which was derived by us from 
<a href="https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery/data">
<b>Semantic segmentation of aerial imagery</b>
</a>
<br><br>
<b>Divide-and-Conquer Strategy</b><br>
Since the images and masks of the Aerial-Imagery are very large (2K to 5K pixels),
we adopted the following <b>Divide-and-Conquer Strategy</b> for building our segmentation model.
<br>
<br>
Please see also our experiment <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Augmented-IDRiD">TensorFlow-FlexUNet-Tiled-Image-Segmentation-Augmented-IDRiD</a>
<br>
<br>
<b>1. Non-Tiled dataset</b><br>
We generated a Non-Tiled dataset by combining all images and masks in  each <b>Tile </b>folder
in the <b>Semantic segmentation of aerial imagery</b>.
<br><br>
<b>2. Augmented-Tiled Image Mask Dataset</b><br>
We generated a PNG image and mask datasets of 512x512 pixels tiledly-split dataset from
the Non-Tiled dataset by an offline augmentation tool.
<br>
<br>
<b>3. Train Model by Tiled ImageMask Dataset</b><br>
We trained and validated the TensorFlowFlexUNet model for Aerial-Imagery by using the Augmented-Tiled-Aerial-Imagery dataset.
<br><br>
<b>4. Tiled Image Segmentation</b><br>
We applied our Tiled-Image Segmentation inference method to predict mask regions for the mini_test images with a resolution of 2K to 5K pixels.
<br><br>
<hr>
<b>Actual Image Segmentation for Non-Tiled Original Images of 2K to 4K pixels</b><br>
As shown below, the tiled_inferred masks predicted by our segmentation model trained on the 
Augmented Tiled dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<b>rgb_map(Building:#3C1098, Unpaved area:#8429F6, Road:#6EC1E4, Vegetation:#FEDD3A, Water:#E2A929, Unlabeled:#9B9B9)</b><br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/images/2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/masks/2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test_output_tiled/2.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/images/4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/masks/4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test_output_tiled/4.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/images/6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/masks/6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test_output_tiled/6.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
1 Dataset Citation
</h3>
The dataset used here was derived from 
<a href="https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery/data">
<b>Semantic segmentation of aerial imagery</b>
</a>
<br>
Satellite images of Dubai, the UAE segmented into 6 classes<br>
<br>
<b>About Dataset</b><br>
<b>Context</b><br>
Humans in the Loop is publishing an open access dataset annotated for a joint project with the Mohammed Bin Rashid Space Center in Dubai, the UAE.
<br>
<br>
<b>Content</b><br>

The dataset consists of aerial imagery of Dubai obtained by MBRSC satellites and annotated with pixel-wise semantic segmentation in 6 classes. The total volume of the dataset is 72 images grouped into 6 larger tiles. 
The classes (added RGB row by T.Arai) are :
<br><br>
<table border=1 style="border-collapse: collapse;">
<tr><th>Index</th><th>Category</th><th>Color</th><th>RGB</th><tr>
<tr><td>1</td><td>Building</td><td>#3C1098</td>    <td>(60,16,152)</td></tr>
<tr><td>2</td><td>Land (unpaved area)</td><td>#8429F6</td><td>(132,41,246)</td></tr>
<tr><td>3</td><td>Road</td><td>#6EC1E4</td><td>(110,193,228)</td></tr>
<tr><td>4</td><td>Vegetation</td><td>#FEDD3A</td><td>(254,221,58)</td></tr>
<tr><td>5</td><td>Water</td><td> #E2A929</td><td>(226,169,41)</td></tr>
<tr><td>6</td><td>Unlabeled</td><td> #9B9B9</td><td>(155,155,155)</td></tr>
</table>

<br>
<b>Acknowledgements</b><br>
The images were segmented by the trainees of the Roia Foundation in Syria.
<br><br>
<b>License</b><br>
<a href="https://creativecommons.org/publicdomain/zero/1.0/">
CC0: Public Domain
</a>
<br>
<br>
<h3>
2 Aerial-Imagery ImageMask Dataset
</h3>
<h4>2.1 Augmented ImageMask Dataset</h4>
 If you would like to train this Aerial-Imagery Segmentation model by yourself,
 please download <a href="https://drive.google.com/file/d/1cZ9-JmCyJ8jjmErQ2ijvo5y4JT6AZ-TW/view?usp=sharing">
 <b>Augmented-Tiled-Aerial-Imagery-ImageMask-Dataset.zip.zip </b></a>
on the google drive, expand the downloaded, and put it under dataset folder to be:
<pre>
./dataset
└─Aerial-Imagery
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>Aerial-Imagery Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Aerial-Imagery/Aerial-Imagery_Statistics.png" width="512" height="auto"><br>
<br>

As shown above, the number of images of train and valid datasets is not so large to use for a training set of our segmentation model.
<br><br>
<h4>2.2 ImageMask Dataset Derivation</h4>
The original folder structure of <b>Semantic segmentation of aerial imagery</b> is the following.
<pre>
./
├─Tile 1
│  ├─images
│  │  ├─image_part_001.jpg
...
│  │  └─image_part_009.jpg
│  └─masks
│  │  ├─image_part_001.png
...
│  │  └─image_part_009.png
...
└─Tile 8
   ├─images
   │  ├─image_part_001.jpg
...
   │  └─image_part_009.jpg
   └─masks
      ├─image_part_001.png
...
      └─image_part_009.png
</pre>
As shown above, each <b>Tile</b> folder contains 9 images and their corresponding masks, which are various pixel size.
Hence we re-generated the original Non-Tiled 9 images and 9 mask files from the 9 Tile folders,
and then split the Non-Tiled images and masks into 512x512 pixels tiles, which can be acceptable image size to 
train our segmentation model<br>

We used the following 3 Python scripts to generate Augmented-Tiled-Aerial-Imagery-ImageMask-Dataset.<br>
<ul>
    <li><a href="./generator/Preprocessor.py">Preprocessor.py</a></li>
    <li><a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a></li>
    <li><a href="./generator/split_master.py">split_master.py</a></li>
</ul>

(1) Firstly, we generated a <b>Non-Tiled-Aerial-Imagery</b> dataset by combining all images and masks in each <b>Tile </b>folder
in the <b>Semantic segmentation of aerial imagery</b> to a single large image and mask files respectively, 
 by using <a href="./generator/Preprocessor.py">Preprocessor.py</a>
 <br><br>
 <b>Combined Non-Tiled-Images</b><br>
 <img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/asset/Non-Tiled-Images.png" width="1024" height="auto"><br>
 <br>
  <b>Combined Non-Tiled-Masks</b><br>
 <img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/asset/Non-Tiled-Masks.png" width="1024" height="auto"><br>
 <br>
(2) Secondly, we generated an <b>Augmented-Tiled-Aerial-Imagery-master</b> dataset from the 9 images/masks of<b>Non-Tiled-Aerial-Imagery</b>
by using<a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a>
, which splits the large images and masks of Non-Tiled dataset into a lot of 512x512 pixels small tiles and augments
the tiles. 
<br><br>
(3) Finally, we generated an <b>Augmented-Tiled-Aerial-Imagery-ImageMask-Dataset</b> from 
the <b>Augmented-Tiled-Aerial-Imagery-master</b> 
by using <a href="./generator/split_master.py">split_master.py</a>, which simply split the master into train, valid and test subsets.
<br><br>
<h4>2.3 Train images and masks sample</h4>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowUNet Model
</h3>
 We trained Aerial-Imagery TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Aerial-Imagery/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Aerial-Imagery and, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 2

base_filters   = 16
base_kernels   = (7,7)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Aerial-Imagery 1+1 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
; Hex color
;               Building:#3C1098, Unpaved area:#8429F6, Road:#6EC1E4, Vegetation:#FEDD3A, Water:#E2A929, Unlabeled:#9B9B9
;                      Building      Unpaved area    Road             Vegetation,     Water            Unlabeled
rgb_map = {(0,0,0):0, (60,16,152):1,(132,41,246):2, (110,193,228):3, (254,221,58):4, (226,169,41):5, (155,155,155):6}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 38,39,40)</b><br>
<img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 77,78,79)</b><br>
<img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was stopped at epoch 79 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/asset/train_console_output_at_epoch79.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Aerial-Imagery/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Aerial-Imagery/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Aerial-Imagery</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Aerial-Imagery.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/asset/evaluate_console_output_at_epoch79.png" width="720" height="auto">
<br><br>Image-Segmentation-Aerial-Imagery

<a href="./projects/TensorFlowFlexUNet/Aerial-Imagery/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this Aerial-Imagery/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.3603
dice_coef_multiclass,0.8587
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Aerial-Imagery</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Aerial-Imagery.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Tiled_inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Non-Tiled Images of 2K to 5K pixels </b><br>
As shown below, the tiled_inferred masks predicted by our segmentation model trained on the 
Augmented Tiled dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<b>rgb_map(Building:#3C1098, Unpaved area:#8429F6, Road:#6EC1E4, Vegetation:#FEDD3A, Water:#E2A929, Unlabeled:#9B9B9)</b><br>
<br>
<table>
<tr>

<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/images/2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/masks/2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test_output_tiled/2.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/images/3.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/masks/3.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test_output_tiled/3.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/images/4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/masks/4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test_output_tiled/4.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/images/5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/masks/5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test_output_tiled/5.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/images/6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/masks/6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test_output_tiled/6.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/images/7.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test/masks/7.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aerial-Imagery/mini_test_output_tiled/7.png" width="320" height="auto"></td>
</tr>


</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Satellite Imagery Aerial-Imagery Segmentation</b><br>
Nithish<br>
<a href="https://medium.com/@nithishmailme/satellite-imagery-road-segmentation-ad2964dc3812">
https://medium.com/@nithishmailme/satellite-imagery-road-segmentation-ad2964dc3812
</a>
<br>
<br>
<b>2. Deep Learning-based Aerial-Imagery Segmentation Using Aerial Images: A Comparative Study</b><br>
Kamal KC, Alaka Acharya, Kushal Devkota, Kalyan Singh Karki, and Surendra Shrestha<br>
<a href="https://www.researchgate.net/publication/382973365_Deep_Learning-based_Aerial-Imagery_Segmentation_Using_Aerial_Images_A_Comparative_Study">
https://www.researchgate.net/publication/382973365_Deep_Learning-based_Aerial-Imagery_Segmentation_Using_Aerial_Images_A_Comparative_Study</a>
<br>
<br>
<b>3. A Comparative Study of Deep Learning Methods for Automated Aerial-Imagery Network<br>
Extraction from High-Spatial-ResolutionRemotely Sensed Imagery</b><br>
Haochen Zhou, Hongjie He, Linlin Xu, Lingfei Ma, Dedong Zhang, Nan Chen, Michael A. Chapman, and Jonathan Li<br>
<a href="https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/documents/march2025_zhou_10.14358_pers_24-00100r2.pdf">
https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/documents/march2025_zhou_10.14358_pers_24-00100r2.pdf
</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>5. TensorFlow-FlexUNet-Tiled-Image-Segmentation-Concrete-Crack</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Concrete-Crack">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Concrete-Crack
</a>
<br>
<br>

