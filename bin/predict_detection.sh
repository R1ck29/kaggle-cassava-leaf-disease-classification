# python -m src.tools.predict +experiment=effdet_person_cosine_annealing \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20200914_20_41_11/

# python -m src.tools.predict +experiment=effdet_person_ca_v2 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20200919_22_05_14/

# python -m src.tools.predict +experiment=effdet_person_ca_v3 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201001_12_26_06/

# python -m src.tools.predict +experiment=effdet_person_ca_v4 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201005_12_47_57/

# python -m src.tools.predict +experiment=effdet_person_ca_v5 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201110_17_14_57/  #20201010_00_40_05/

# python -m src.tools.predict +experiment=effdet_cvat \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201019_22_32_06/

# HYDRA_FULL_ERROR=1 python -m src.tools.predict +experiment=fasterrcnn_model001 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201026_12_42_08_2/

# python -m src.tools.predict +experiment=effdet_person_ca_v6 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201015_18_54_35/ \
#                                       TEST_ID=test_soft_nms \
#                                       TEST.ENSEMBLE_BOXES.NAME=SoftNMS \

# python -m src.tools.predict +experiment=effdet_person_ca_v6 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201015_18_54_35/ \
#                                       TEST_ID=test_nms \
#                                       TEST.ENSEMBLE_BOXES.NAME=NMS

# python -m src.tools.predict +experiment=effdet_person_ca_v6 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201015_18_54_35/ \
#                                       TEST_ID=test_nmw \
#                                       TEST.ENSEMBLE_BOXES.NAME=NMW \

# python -m src.tools.predict +experiment=effdet_person_ca_v6 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201015_18_54_35/
#                                       TEST_ID=test_wbf_2 \
#                                       TEST.ENSEMBLE_BOXES.NAME=WBF \


# HYDRA_FULL_ERROR=1 python -m src.tools.predict +experiment=fasterrcnn_wheat \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201110_17_19_36/


# python -m src.tools.predict +experiment=effdet_person_ca_v5 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201111_16_57_51 #20201111_19_27_07/

# python -m src.tools.predict +experiment=effdet_person_ca_v5 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201010_00_40_05/ \
#                                       TEST_ID=soft_nms \
#                                       TEST.ENSEMBLE_BOXES.NAME=SoftNMS \

# python -m src.tools.predict +experiment=effdet_person_ca_v5 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201010_00_40_05/
#                                       TEST_ID=wbf \
#                                       TEST.ENSEMBLE_BOXES.NAME=WBF \


# python -m src.tools.predict +experiment=effdet_person_ca_v5 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201010_00_40_05/ \
#                                       TEST_ID=nms \
#                                       TEST.ENSEMBLE_BOXES.NAME=NMS

# python -m src.tools.predict +experiment=effdet_person_ca_v5 \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201010_00_40_05/ \
#                                       TEST_ID=nmw \
#                                       TEST.ENSEMBLE_BOXES.NAME=NMW \

# 5 fold version
python -m src.tools.predict +experiment=effdet_person_ca_v5 \
                                      MODEL_PATH=models/$\{MODEL_ID\}/20201218_17_50_07/ \
                                      TEST_ID=soft_nms \
                                      TEST.ENSEMBLE_BOXES.NAME=WBF \
