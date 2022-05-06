# Runing under the project root dir
# Training on the original dataset
python run.py --dataset onto --ontology ./data/ontology/onto_ontology.txt \
--train ./data/ontonotes/g_train.json --valid ./data/ontonotes/g_dev.json \
--test ./data/ontonotes/g_test.json --fc_param ./save/onto_fc.pth \
--cuda --lr 2e-6 --save_dir ./save/original

# Denoising
python denoise.py --dataset onto --ontology ./data/ontology/onto_ontology.txt \
--train ./data/ontonotes/g_train.json --valid ./data/ontonotes/g_dev.json \
--fc_param ./save/onto_fc.pth --save_dir ./save/denoise \
--filter_on --alpha 2.0 --entropy_loss_on --beta 1.0 --threshold 0.3 \
--cuda --lr 2e-6 --eval --stage2_step 5000

# Training on the denoised dataset
python run.py --dataset onto --ontology ./data/ontology/onto_ontology.txt \
--train ./data/ontonotes/g_train.json --valid ./data/ontonotes/g_dev.json \
--test ./data/ontonotes/g_test.json --fc_param ./save/onto_fc.pth \
--cuda --lr 2e-6 --save_dir ./save/test --label_correction \
--FP_mask ./save/denoise/onto/FP_mask.pth --FN_mask ./save/denoise/onto/FN_mask.pth

# Save the denoised dataset
python save_denoised_datasets.py --dataset onto \
--ontology ./data/ontology/onto_ontology.txt --train ./data/ontonotes/g_train.json \
--FP_mask ./save/denoise/onto/FP_mask.pth --FN_mask ./save/denoise/onto/FN_mask.pth \
--save_dir ./data2
