python run.py --dataset ultra --ontology ./data/ontology/types.txt \
--train ./denoised_data/Ultra-fine/train.json --valid ./data/crowd/dev.json \
--test ./data/crowd/test.json --fc_param ./save/ultra_fc.pth \
--cuda --lr 2e-5 --save_dir ./save/test

python run.py --dataset augmented_onto --ontology ./data/ontology/onto_ontology.txt \
--train ./denoised_data/OntoNotes/train.json --valid ./data/ontonotes/g_dev.json \
--test ./data/ontonotes/g_test.json --fc_param ./save/augmented_onto_fc.pth \
--cuda --lr 2e-6 --save_dir ./save/test
