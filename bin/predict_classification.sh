# python -m src.tools.predict +experiment=person_attribute \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201202_22_42_27 \
#                                       DATA.N_FOLD=1

# adamw version
# python -m src.tools.predict +experiment=person_attribute_adamw \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201206_18_46_58 \
#                                       DATA.N_FOLD=1

# 1 albumentations SGD val_loss
python -m src.tools.predict +experiment=person_attribute \
                                      MODEL_PATH=models/$\{MODEL_ID\}/20201207_20_21_34 \
#                                       SYSTEM.GPUS="1"

# PA100k test
# python -m src.tools.predict +experiment=person_attribute_pa100k \
#                             MODEL_PATH=models/$\{MODEL_ID\}/20201202_11_09_22

# albu val_score SGD
# python -m src.tools.predict +experiment=person_attribute \
#                             MODEL_PATH=models/$\{MODEL_ID\}/20201202_22_42_27


# python -m src.tools.predict +experiment=person_attribute_adamw \
#                             MODEL_PATH=models/$\{MODEL_ID\}/20201206_18_46_58

# no albu
# python -m src.tools.predict +experiment=person_attribute \
#                             MODEL_PATH=models/$\{MODEL_ID\}/20201208_11_42_02

# 3
# python -m src.tools.predict +experiment=person_attribute_cosine \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201208_21_07_18

# 2
# python -m src.tools.predict +experiment=person_attribute_cosinewarm \
#                                       MODEL_PATH=models/$\{MODEL_ID\}/20201209_19_48_19

# val score
# python -m src.tools.predict +experiment=person_attribute \
#                             MODEL_PATH=models/$\{MODEL_ID\}/20201210_15_02_36

# val loss
# python -m src.tools.predict +experiment=person_attribute \
#                             MODEL_PATH=models/$\{MODEL_ID\}/20201213_13_33_12
                            # TEST.BEST_WEIGHT_TYPE='val_loss'