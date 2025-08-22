d={
    "ETTh1": {
        "DMM": {
            "MAR": {
                0.2: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1 --model DMM --data ETTh1 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--backbone cnn --batch_size 128 --learning_rate 0.0007 --n_group_inner_layers 2 --input_with_mask 1 "
                    "--d_layers 1 --d_model 512 --kernel_size 1 --n_heads 8 --d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 "
                    "--dropout 0 --kld_weight 1e-4 ",
                0.4: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1 --model DMM --data ETTh1 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--backbone attn --batch_size 128 --learning_rate 0.001 --n_group_inner_layers 2 --input_with_mask 1 "
                    "--d_layers 1 --d_model 512 --kernel_size 1 --n_heads 8 --d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 "
                    "--dropout 0 --kld_weight 1e-4 ",
                0.6: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1 --model DMM --data ETTh1 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--backbone cnn --batch_size 512 --learning_rate 0.0007 --n_group_inner_layers 2 --input_with_mask 1 "
                    "--d_layers 1 --d_model 512 --kernel_size 1 --n_heads 8 --d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 "
                    "--kld_weight 1e-5 ",
            },
            "MNAR": {
                0.2: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1 --model DMM --data ETTh1 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone attn --batch_size 256 --learning_rate 1e-2 --n_group_inner_layers 2 --input_with_mask 1 "
                    "--d_layers 1 --d_model 256 --kernel_size 1 --n_heads 10 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 --kld_weight 1e-2 ",
                0.4: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1 --model DMM --data ETTh1 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone attn --batch_size 128 --learning_rate 1e-3 --n_group_inner_layers 2 --input_with_mask 1 "
                    "--d_layers 1 --d_model 256 --kernel_size 1 --n_heads 10 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 --kld_weight 1e-3 ",
                0.6: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1 --model DMM --data ETTh1 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone attn --batch_size 128 --learning_rate 5e-4 --n_group_inner_layers 2 --input_with_mask 1 "
                    "--d_layers 1 --d_model 256 --kernel_size 1 --n_heads 10 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 --kld_weight 1e-2 ",
            },
        },
    },
    "ETTh2": {
        "DMM": {
            "MAR": {
                0.2: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2 --model DMM --data ETTh2 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 512 --learning_rate 0.0001 --n_group_inner_layers 2 --input_with_mask 1 --d_layers 1 --d_model 512 --kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 --kld_weight 1e-4 ",
                0.4: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2 --model DMM --data ETTh2 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type1 --backbone attn --batch_size 128 --learning_rate 1e-2 --n_group_inner_layers 2 --input_with_mask 1 "
                    "--d_layers 1 --d_model 512 --kernel_size 1 --n_heads 10 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 --kld_weight 1e-2 ",
                0.6: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2 --model DMM --data ETTh2 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 512 --learning_rate 1e-4 --n_group_inner_layers 2 --input_with_mask 1 --d_layers 1 --d_model 512 --kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 --kld_weight 1e-4 ",
            },
            "MNAR": {
                0.2: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2 --model DMM --data ETTh2 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 128 --learning_rate 0.0006 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 "
                    "--d_model 256 --kernel_size 1 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
                0.4: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2 --model DMM --data ETTh2 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 512 --learning_rate 0.0006 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 "
                    "--d_model 512 --kernel_size 1 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
                0.6: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2 --model DMM --data ETTh2 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 256 --learning_rate 0.001 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 "
                    "--d_model 512 --kernel_size 1 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
            },
        },
    },
    "ETTm1": {
        "DMM": {
            "MAR": {
                0.2: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1 --model DMM --data ETTm1 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type1 --backbone attn --batch_size 128 --learning_rate 1e-5 --n_group_inner_layers 1 --input_with_mask 1 "
                    "--d_layers 1 --d_model 256 --kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
                0.4: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1 --model DMM --data ETTm1 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone attn --batch_size 64 --learning_rate 1e-4 --n_group_inner_layers 2 --input_with_mask 1 "
                    "--d_layers 1 --d_model 512 --kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
                0.6: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1 --model DMM --data ETTm1 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type1 --backbone attn --batch_size 128 --learning_rate 1e-5 --n_group_inner_layers 1 --input_with_mask 1 "
                    "--d_layers 1 --d_model 256 --kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
            },
            "MNAR": {
                0.2: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1 --model DMM --data ETTm1 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone attn --batch_size 256 --learning_rate 6e-4 --n_group_inner_layers 2 --input_with_mask 1 "
                    "--d_layers 1 --d_model 256 --kernel_size 1 --n_heads 10 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 --kld_weight 1e-3 ",
                0.4: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1 --model DMM --data ETTm1 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone attn --batch_size 64 --learning_rate 1e-4 --n_group_inner_layers 2 --input_with_mask 1 "
                    "--d_layers 1 --d_model 512 --kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 --kld_weight 1e-3 ",
                0.6: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1 --model DMM --data ETTm1 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone attn --batch_size 256 --learning_rate 6e-4 --n_group_inner_layers 2 --input_with_mask 1 "
                    "--d_layers 1 --d_model 256 --kernel_size 1 --n_heads 10 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 --kld_weight 1e-3 ",
            },
        },
    },
    "ETTm2": {
        "DMM": {
            "MAR": {
                0.2: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --model_id ETTm2 --model DMM --data ETTm2 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 512 --learning_rate 5e-4 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 --d_model 512 "
                    "--kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 --kld_weight 1e-5 --sparsity_weight 1e-3 ",
                0.4: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --model_id ETTm2 --model DMM --data ETTm2 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 512 --learning_rate 7e-4 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 --d_model 512 "
                    "--kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 --kld_weight 1e-4 --sparsity_weight 1e-2 ",
                0.6: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --model_id ETTm2 --model DMM --data ETTm2 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 64 --learning_rate 3e-3 --emb_dim 128 --patch_size 4 --n_group_inner_layers 1 --input_with_mask 1 "
                    "--d_layers 1 --d_model 512 --kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 --kld_weight 1e-3",
            },
            "MNAR": {
                0.2: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --model_id ETTm2 --model DMM --data ETTm2 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 512 --learning_rate 5e-4 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 --d_model 512 "
                    "--kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
                0.4: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --model_id ETTm2 --model DMM --data ETTm2 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 512 --learning_rate 1e-5 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 --d_model 512 "
                    "--kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
                0.6: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --model_id ETTm2 --model DMM --data ETTm2 --features M --seq_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 512 --learning_rate 1e-5 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 "
                    "--d_model 512 --emb_dim 256 --patch_size 8 --kld_weight 1e-5 "
                    "--kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 7 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
            },
        },
    },
    "Exchange": {
        "DMM": {
            "MAR": {
                0.2: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv --model_id Exchange --model DMM --data custom --features M --seq_len 96 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 128 --learning_rate 0.002 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 --d_model 512 --kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 8 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
                0.4: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv --model_id Exchange --model DMM --data custom --features M --seq_len 96 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 128 --learning_rate 0.0006 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 --d_model 512 --kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 8 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
                0.6: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv --model_id Exchange --model DMM --data custom --features M --seq_len 96 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 128 --learning_rate 0.002 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 --d_model 512 --kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 8 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
            },
            "MNAR": {
                0.2: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv --model_id Exchange --model DMM --data custom --features M --seq_len 96 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 64 --learning_rate 0.005 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 --d_model 256 "
                    "--emb_dim 128 --patch_size 4 --kernel_size 1 --n_heads 8 --kld_weight 1e-4 "
                    "--d_inner 128 --Z_dim 8 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
                0.4: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv --model_id Exchange --model DMM --data custom --features M --seq_len 96 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 128 --learning_rate 0.003 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 --d_model 512 "
                    "--kernel_size 1 --n_heads 8 "
                    "--d_inner 128 --Z_dim 8 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
                0.6: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv --model_id Exchange --model DMM --data custom --features M --seq_len 96 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 64 --learning_rate 0.001 --n_group_inner_layers 1 --input_with_mask 1 --d_layers 1 --d_model 256 "
                    "--emb_dim 128 --patch_size 4 --kernel_size 1 --n_heads 8 --kld_weight 1e-4 "
                    "--d_inner 128 --Z_dim 8 --z_dim 4 --d_k 128 --d_v 128 --dropout 0 ",
            },
        },
    },
    "Weather": {
        "DMM": {
            "MAR": {
                0.2: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather --model DMM --data custom --features M --seq_len 96 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --itr 1 "
                    "--lradj type1 --backbone cnn --batch_size 64 --learning_rate 5e-5 --d_model 256 "
                    "--Z_dim 21 --z_dim 11 ",
                0.4: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather --model DMM --data custom --features M --seq_len 96 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --itr 1 "
                    "--lradj type1 --backbone cnn --batch_size 64 --learning_rate 1e-5 --d_layers 1 --d_model 256 "
                    "--Z_dim 21 --z_dim 11 ",
                0.6: "python -u run.py --mask_type MAR --task_name imputation --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather --model DMM --data custom --features M --seq_len 96 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --itr 1 "
                    "--lradj type1 --backbone cnn --batch_size 64 --learning_rate 1e-5 --d_layers 1 --d_model 256 "
                    "--Z_dim 21 --z_dim 11 ",
            },
            "MNAR": {
                0.2: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather --model DMM --data custom --features M --seq_len 96 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --itr 1 "
                    "--lradj type1 --backbone cnn --batch_size 512 --learning_rate 3e-3 --d_layers 1 "
                    "--d_model 256 --emb_dim 256 --patch_size 8 --kld_weight 1e-5 "
                    "--Z_dim 21 --z_dim 11 ",
                0.4: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather --model DMM --data custom --features M --seq_len 96 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --itr 1 "
                    "--lradj type0 --backbone cnn --batch_size 512 --learning_rate 1e-3 --d_layers 1 "
                    "--d_model 512 --emb_dim 256 --patch_size 8 --kld_weight 1e-5 "
                    "--Z_dim 21 --z_dim 11 ",
                0.6: "python -u run.py --mask_type MNAR --task_name imputation --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather --model DMM --data custom --features M --seq_len 96 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --itr 1 "
                    "--lradj type1 --backbone cnn --batch_size 512 --learning_rate 0.005 --d_layers 1 "
                    "--d_model 256 --emb_dim 256 --patch_size 8 --kld_weight 1e-5 "
                    "--Z_dim 21 --z_dim 11 ",
            },
        },
    },
    }