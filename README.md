# FISH SEGMENTATION

`cd segmentation/; git clone https://github.com/alzayats/DeepFish.git`

# COLOR STUDY

A collection of methods to analyse color changes in species of fish with the interest to extend it to other species of interest that show variations in color as parts of different aspects of their evolution! 

![Cherry-picking](resources/sample_results.png?raw=true "Sample GMM Color Distributions")

### NOTE: The number of gaussians here are taken to be more than the number of colors on purpose to enable distribution based averaging after the hierarchical segmentation of organs of the fish!

#TODO: Add illumination correction to GMM to eliminate it in favor of color
https://stackoverflow.com/questions/63933790/robust-algorithm-to-detect-uneven-illumination-in-images-detection-only-needed

Link to result images from GMM-based color segmentation: https://drive.google.com/drive/u/0/folders/1xanvOQpJXh8iS_iCdUrQ5MeFQQbP6gly

#### CURRENT DATASETS (https://drive.google.com/drive/folders/1Jecz9_nhU0kngk2iSdJKmcHYiMJRCZzt):
#### https://huggingface.co/datasets/hans0812/fish-composite-segmentation/upload/main

##### Please feel free to suggest any new ones involving fish (hansk@nyu.edu)

##### Make sure the folder names are unique!

 - Folder: Machine learning training set/photos 1.30.2019/original image/ #Images: 92
 - Folder: Machine learning training set/photos 1.30.2019/eye/ #Images: 91
 - Folder: Machine learning training set/photos 1.30.2019/pectoral fin/ #Images: 91
 - Folder: Machine learning training set/photos 1.30.2019/pelvic fin/ #Images: 91
 - Folder: Machine learning training set/photos 1.30.2019/dorsal fin/ #Images: 91
 - Folder: Machine learning training set/photos 1.30.2019/operculum/ #Images: 91
 - Folder: Machine learning training set/photos 1.30.2019/humeral blotch/ #Images: 91
 - Folder: Machine learning training set/photos 1.30.2019/ventral side/ #Images: 91
 - Folder: Machine learning training set/photos 1.30.2019/whole body/ #Images: 91
 - Folder: Machine learning training set/photos 1.30.2019/dorsal side/ #Images: 91
 - Folder: Machine learning training set/photos 1.30.2019/head/ #Images: 91
 - Folder: Machine learning training set/photos 1.30.2019/anal fin/ #Images: 91
 - Folder: Machine learning training set/photos 1.30.2019/caudal fin/ #Images: 91
 - Folder: Machine learning training set/T1/original image/ #Images: 9
 - Folder: Machine learning training set/T1/eye/ #Images: 9
 - Folder: Machine learning training set/T1/pectoral fin/ #Images: 9
 - Folder: Machine learning training set/T1/pelvic fin/ #Images: 9
 - Folder: Machine learning training set/T1/dorsal fin/ #Images: 9
 - Folder: Machine learning training set/T1/operculum/ #Images: 9
 - Folder: Machine learning training set/T1/humeral blotch/ #Images: 9
 - Folder: Machine learning training set/T1/ventral side/ #Images: 9
 - Folder: Machine learning training set/T1/whole body/ #Images: 9
 - Folder: Machine learning training set/T1/dorsal side/ #Images: 9
 - Folder: Machine learning training set/T1/head/ #Images: 9
 - Folder: Machine learning training set/T1/anal fin/ #Images: 9
 - Folder: Machine learning training set/T1/caudal fin/ #Images: 9
 - Folder: Machine learning training set/Fish Photography that needs to be matched for HK/2.26.2020/ #Images: 18
 - Folder: Machine learning training set/Fish Photography that needs to be matched for HK/3.11.2020/ #Images: 19
 - Folder: Machine learning training set/Fish Photography that needs to be matched for HK/3.04.2020/ #Images: 18
 - Folder: Machine learning training set/Fish Photography that needs to be matched for HK/2.5.20 fish/ #Images: 17
 - Folder: Machine learning training set/Fish Photography that needs to be matched for HK/2.19.2020/ #Images: 13
 - Folder: Machine learning training set/Fish Photography that needs to be matched for HK/01.15.20 fish/ #Images: 1
 - Folder: Machine learning training set/Phase 2 Color quantification for Hans/Yellow/ #Images: 36
 - Folder: Machine learning training set/Phase 2 Color quantification for Hans/Blue/ #Images: 41
 - Folder: Machine learning training set/T0/original image/ #Images: 15
 - Folder: Machine learning training set/T0/eye/ #Images: 15
 - Folder: Machine learning training set/T0/pectoral fin/ #Images: 15
 - Folder: Machine learning training set/T0/pelvic fin/ #Images: 15
 - Folder: Machine learning training set/T0/dorsal fin/ #Images: 15
 - Folder: Machine learning training set/T0/operculum/ #Images: 15
 - Folder: Machine learning training set/T0/humeral blotch/ #Images: 15
 - Folder: Machine learning training set/T0/ventral side/ #Images: 15
 - Folder: Machine learning training set/T0/whole body/ #Images: 15
 - Folder: Machine learning training set/T0/dorsal side/ #Images: 15
 - Folder: Machine learning training set/T0/head/ #Images: 15
 - Folder: Machine learning training set/T0/anal fin/ #Images: 15
 - Folder: Machine learning training set/T0/caudal fin/ #Images: 15
 - Folder: Machine learning training set/12-23-2019/original image/ #Images: 596
 - Folder: Machine learning training set/12-23-2019/eye/ #Images: 580
 - Folder: Machine learning training set/12-23-2019/pectoral fin/ #Images: 579
 - Folder: Machine learning training set/12-23-2019/pelvic fin/ #Images: 580
 - Folder: Machine learning training set/12-23-2019/dorsal fin/ #Images: 580
 - Folder: Machine learning training set/12-23-2019/operculum/ #Images: 585
 - Folder: Machine learning training set/12-23-2019/humeral blotch/ #Images: 580
 - Folder: Machine learning training set/12-23-2019/ventral side/ #Images: 580
 - Folder: Machine learning training set/12-23-2019/whole body/ #Images: 580
 - Folder: Machine learning training set/12-23-2019/dorsal side/ #Images: 580
 - Folder: Machine learning training set/12-23-2019/head/ #Images: 580
 - Folder: Machine learning training set/12-23-2019/anal fin/ #Images: 580
 - Folder: Machine learning training set/12-23-2019/caudal fin/ #Images: 580
 - Folder: Machine learning training set/12-23-2019/stripes/ #Images: 263
 - Folder: deepfish/zip/fish_tray_images_2021_06-09/ #Images: 158
 - Folder: deepfish/zip/fish_tray_json_labels/ #Images: 1
 - Folder: deepfish/zip/fish_tray_images_2021_04_20/ #Images: 277
 - Folder: deepfish/zip/fish_tray_images_2021_04_01/ #Images: 209
 - Folder: deepfish/zip/fish_tray_images_2021_05_01/ #Images: 328
 - Folder: deepfish/zip/fish_tray_images_2021_05_17/ #Images: 352
 - Folder: DeepFish/Localization/images/valid/ #Images: 1600
 - Folder: DeepFish/Localization/images/empty/ #Images: 1600
 - Folder: DeepFish/Classification/9852/valid/ #Images: 221
 - Folder: DeepFish/Classification/9852/empty/ #Images: 254
 - Folder: DeepFish/Classification/7426/valid/ #Images: 714
 - Folder: DeepFish/Classification/7426/empty/ #Images: 3304
 - Folder: DeepFish/Classification/9870/valid/ #Images: 190
 - Folder: DeepFish/Classification/9870/empty/ #Images: 174
 - Folder: DeepFish/Classification/7434/valid/ #Images: 1141
 - Folder: DeepFish/Classification/7434/empty/ #Images: 1654
 - Folder: DeepFish/Classification/7585/valid/ #Images: 949
 - Folder: DeepFish/Classification/7585/empty/ #Images: 1190
 - Folder: DeepFish/Classification/9898/valid/ #Images: 117
 - Folder: DeepFish/Classification/9898/empty/ #Images: 443
 - Folder: DeepFish/Classification/7482/valid/ #Images: 1461
 - Folder: DeepFish/Classification/7482/empty/ #Images: 3516
 - Folder: DeepFish/Classification/9892/valid/ #Images: 166
 - Folder: DeepFish/Classification/9892/empty/ #Images: 360
 - Folder: DeepFish/Classification/7117/valid/ #Images: 2029
 - Folder: DeepFish/Classification/7117/empty/ #Images: 90
 - Folder: DeepFish/Classification/9862/valid/ #Images: 123
 - Folder: DeepFish/Classification/9862/empty/ #Images: 311
 - Folder: DeepFish/Classification/9908/valid/ #Images: 298
 - Folder: DeepFish/Classification/9908/empty/ #Images: 602
 - Folder: DeepFish/Classification/9894/valid/ #Images: 244
 - Folder: DeepFish/Classification/9894/empty/ #Images: 1298
 - Folder: DeepFish/Classification/7393/valid/ #Images: 547
 - Folder: DeepFish/Classification/7393/empty/ #Images: 1554
 - Folder: DeepFish/Classification/7268/valid/ #Images: 1467
 - Folder: DeepFish/Classification/7398/valid/ #Images: 1704
 - Folder: DeepFish/Classification/7398/empty/ #Images: 814
 - Folder: DeepFish/Classification/7463/valid/ #Images: 1790
 - Folder: DeepFish/Classification/7463/empty/ #Images: 1465
 - Folder: DeepFish/Classification/9866/valid/ #Images: 855
 - Folder: DeepFish/Classification/9866/empty/ #Images: 262
 - Folder: DeepFish/Classification/9907/valid/ #Images: 394
 - Folder: DeepFish/Classification/9907/empty/ #Images: 1454
 - Folder: DeepFish/Classification/7490/valid/ #Images: 1903
 - Folder: DeepFish/Classification/7490/empty/ #Images: 411
 - Folder: DeepFish/Classification/7623/valid/ #Images: 1096
 - Folder: DeepFish/Classification/7623/empty/ #Images: 1557
 - Folder: DeepFish/Segmentation/images/valid/ #Images: 310
 - Folder: DeepFish/Segmentation/images/empty/ #Images: 310
 - Folder: Fish-Pak/Rohu/body/ #Images: 73
 - Folder: Fish-Pak/Rohu/head/ #Images: 114
 - Folder: Fish-Pak/Rohu/scales/ #Images: 62
 - Folder: Fish-Pak/Cyprinus carpio/Scales/ #Images: 44
 - Folder: Fish-Pak/Cyprinus carpio/Body/ #Images: 50
 - Folder: Fish-Pak/Cyprinus carpio/Head/ #Images: 64
 - Folder: Fish-Pak/Grass Carp/body/ #Images: 11
 - Folder: Fish-Pak/Grass Carp/head/ #Images: 16
 - Folder: Fish-Pak/Grass Carp/scales/ #Images: 9
 - Folder: Fish-Pak/Catla/Scales/ #Images: 11
 - Folder: Fish-Pak/Catla/Body/ #Images: 20
 - Folder: Fish-Pak/Catla/Head/ #Images: 25
 - Folder: pawsey/FDFML/frames/ #Images: 64385
 - Folder: Fish_automated_identification_and_counting/luderick-seagrass/ #Images: 4280
 - Folder: Fish-Pak/Rohu/body #Images: 73
 - Folder: Fish-Pak/Rohu/head #Images: 114
 - Folder: Fish-Pak/Rohu/scales #Images: 62
 - Folder: Fish-Pak/Cyprinus carpio/Scales #Images: 44
 - Folder: Fish-Pak/Cyprinus carpio/Body #Images: 50
 - Folder: Fish-Pak/Cyprinus carpio/Head #Images: 64
 - Folder: Fish-Pak/Grass Carp/body #Images: 11
 - Folder: Fish-Pak/Grass Carp/head #Images: 16
 - Folder: Fish-Pak/Grass Carp/scales #Images: 9
 - Folder: Fish-Pak/Catla/Scales #Images: 11
 - Folder: Fish-Pak/Catla/Body #Images: 20
 - Folder: Fish-Pak/Catla/Head #Images: 25
 - Folder: roboflow/Aquarium-Combined-3/train/images/train/ #Images: 4480
 - Folder: roboflow/Aquarium-Combined-3/valid/images/valid/ #Images: 127
 - Folder: roboflow/Brackish-Underwater-2/train/images/train/ #Images: 11739
 - Folder: roboflow/Brackish-Underwater-2/valid/images/valid/ #Images: 1468
 - Folder: roboflow/Fish-43/train/images/train/ #Images: 952
 - Folder: roboflow/Fish-43/valid/images/valid/ #Images: 272
 - Folder: SUIM/SUIM/train_val/images/ #Images: 1525
 - Folder: SUIM/SUIM/TEST/images/ #Images: 110
 - Folder: foid/foid_images_v100/images/ #Images: 143818
 - Folder: foid/foid_images_v020/images/ #Images: 86029

# TOTAL: 372932 images!
