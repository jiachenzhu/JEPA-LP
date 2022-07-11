import os
import yaml

i = 1
for projector_type in ['1brl']:
    for projector_dims in [[64, 128, 128]]:
        for latent_dim in [256, 512]:
            for hidden_dim in [1024]:
                for num_layers in [4]:
                    for num_steps in [1, 2, 4, 8]:
                        for sim_coeff in [25.0 / 4, 25.0 / 2, 25.0]:
                            for std_coeff in [25.0 / 4, 25.0 / 2, 25.0]:
                                for start_lr in [0.15, 0.3, 0.6, 1.2, 2.4]:
                                    comment = f"JEPA_LP_{i}_{projector_type}_{projector_dims}_{latent_dim}_{hidden_dim}_{num_layers}_{num_steps}_sim_coeff_{sim_coeff}_std_coeff_{std_coeff}_lr_{start_lr}"
                                    comment = comment.replace(" ", "").replace("[", "").replace("]", "").replace(",", "_").replace("'", "")
                                    config = {
                                        "comment": comment,
                                        "projector_type": projector_type,
                                        "projector_dims": projector_dims,
                                        "latent_dim": latent_dim,
                                        "hidden_dim": hidden_dim,
                                        "num_layers": num_layers,
                                        "num_steps": num_steps,
                                        "sim_coeff": sim_coeff,
                                        "std_coeff": std_coeff,
                                        "start_lr": start_lr,
                                    }
                                    with open(os.path.join("configs", f"experiment_{i}.yml"), 'w') as conf_file:
                                        yaml.dump(config, conf_file)
                                
                                    i += 1