# python -m src.tools.train +experiment=effdet_person_cosine_annealing

# python -m src.tools.train +experiment=effdet_person_ca_v3

# python -m src.tools.train +experiment=effdet_person_ca_v4

# python -m src.tools.train +experiment=effdet_person_ca_v5 \
#                             +TRAINER.amp_level='O1' \
#                             +TRAINER.precision=16

# python -m src.tools.train +experiment=effdet_person_ca_v6 \
#                             +TRAINER.amp_level='O1' \
#                             +TRAINER.precision=16

# python -m src.tools.train +experiment=effdet_cvat

python -m src.tools.train +experiment=fasterrcnn_model001