# wheat data test
# python -m src.preprocess.detection.build_features +experiment=effdet_wheat
# python -m src.preprocess.detection.make_folds +experiment=effdet_wheat

# cvat test
# python -m src.preprocess.detection.xml_to_csv +experiment=effdet_cvat
# python -m src.preprocess.detection.build_features +experiment=effdet_cvat
# python -m src.preprocess.detection.train_test_split +experiment=effdet_cvat
# python -m src.preprocess.detection.make_folds +experiment=effdet_cvat

# open images test
# python -m src.preprocess.detection.build_features +experiment=fasterrcnn_model001 #effdet_open_images_person
python -m src.preprocess.detection.make_folds +experiment=effdet_person_ca_v5 #fasterrcnn_model001 #effdet_open_images_person

# for CVAT GT 
# ffmpeg -i /data1/r_suzuki/pipeline/BaseModel/data/person_attribute_demo_4/raw/videos/video_3.mp4 -start_number 0 -b:v 10000k -vsync 0 -an -y -q:v 2 /data1/r_suzuki/pipeline/BaseModel/data/person_attribute_demo_4/raw/images/%05d.jpg

#person attribute demo from CVAT
# python -m src.preprocess.detection.xml_to_csv data=detection \
#                                               model=detection/efficientdet \
#                                               loss=wrmsse \
#                                               test=detection \
#                                               +experiment=effdet_cvat \
#                                               DATA.DATA_ID=person_attribute_demo_4 \
#                                               DATA.TRAIN_IMAGE_DIR=data/$\{DATA.DATA_ID\}/raw/images \
#                                               DATA.CSV_PATH=data/$\{DATA.DATA_ID\}/split/train_folds_cvat_person.csv \
#                                               MODEL_ID=effdet_cvat_person_attribute


# python -m src.preprocess.detection.build_features data=detection \
#                                                   model=detection/efficientdet \
#                                                   loss=wrmsse \
#                                                   test=detection \
#                                                   +experiment=effdet_cvat \
#                                                   DATA.DATA_ID=person_attribute_demo_4 \
#                                                   DATA.TRAIN_IMAGE_DIR=data/$\{DATA.DATA_ID\}/raw/images \
#                                                   DATA.CSV_PATH=data/$\{DATA.DATA_ID\}/split/train_folds_cvat_person.csv \
#                                                   MODEL_ID=effdet_cvat_person_attribute
                                              
# python -m src.preprocess.detection.train_test_split +experiment=effdet_cvat
# python -m src.preprocess.detection.make_folds +experiment=effdet_cvat