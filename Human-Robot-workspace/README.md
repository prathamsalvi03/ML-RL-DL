safety_robot_arm/
├── docker/
   ├── mujoco.Dockerfile
│   ├── ros2.Dockerfile
│   ├── ml.Dockerfile
│   ├── docker-compose.yml
│   └── entrypoints/
│       ├── sim_entry.sh
│       ├── ros2_entry.sh
│       └── train_entry.sh
│
├── mujoco_sim/
│   ├── sim_env.py
│   ├── scene_setup.py
│   └── human_sim.py
│
├── ros2_ws/
│   ├── src/
│   │   ├── perception_node/
│   │   ├── control_node/
│   │   └── intrusion_predictor/
│   └── launch/
│       └── start.launch.py
│
├── ml_training/
│   ├── train_intrusion_model.py
│   ├── evaluate_model.py
│   └── inference.py
│
├── data/
│   ├── raw/                     # raw camera and sim data
│   ├── processed/               # preprocessed tensors
│   └── logs/                    # metrics, events
│
├── models/
│   └── trained/                 # stored DVC-tracked models
│
├── .dvc/                        # DVC config
├── mlflow/                     # MLflow or wandb setup
├── Makefile                    # CLI tasks: make run, train, etc.
└── README.md
