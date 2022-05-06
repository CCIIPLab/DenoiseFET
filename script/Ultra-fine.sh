# Runing under the project root dir
# Training on the original dataset
python run.py --dataset ultra --ontology ./data/ontology/types.txt \
--train ./data/crowd/train.json --valid ./data/crowd/dev.json \
--test ./data/crowd/test.json --fc_param ./save/ultra_fc.pth \
--cuda --lr 2e-5 --save_dir ./save/original

# Denoising
python denoise.py --dataset ultra --ontology ./data/ontology/types.txt \
--train ./data/crowd/train.json --valid ./data/crowd/dev.json \
--fc_param ./save/ultra_fc.pth --save_dir ./save/denoise \
--filter_on --alpha 2.0 --entropy_loss_on --beta 0.5 --threshold 0.1 \
--cuda --lr 2e-5 --stage1_step 1750 --stage2_step 2000

# Training on the denoised dataset
python run.py --dataset ultra --ontology ./data/ontology/types.txt \
--train ./data/crowd/train.json --valid ./data/crowd/dev.json \
--test ./data/crowd/test.json --fc_param ./save/ultra_fc.pth \
--cuda --lr 2e-5 --save_dir ./save/test --label_correction \
--FP_mask ./save/denoise/ultra/FP_mask.pth --FN_mask ./save/denoise/ultra/FN_mask.pth

# Save the denoised dataset
python save_denoised_datasets.py --dataset ultra \
--ontology ./data/ontology/types.txt --train ./data/crowd/train.json \
--FP_mask ./save/denoise/ultra/FP_mask.pth --FN_mask ./save/denoise/ultra/FN_mask.pth \
--save_dir ./data2
