python main.py --c_dim 6 --dataset market --mode train --image_dir ../data/market/bounding_box_train --log_dir ./market/logs \
    --model_save_dir ./market/models --sample_dir ./market/samples --result_dir ../data/market/bounding_box_train_camstyle_stargan4reid

python main.py --c_dim 6 --dataset market --mode test --image_dir ../data/market/bounding_box_train --log_dir ./market/logs \
    --test_iters 220000 --model_save_dir ./market/models --sample_dir ./market/samples --result_dir ../data/market/bounding_box_train_camstyle_stargan4reid