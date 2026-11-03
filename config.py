GRID_SIZE = 30

N_FOOD = 18
N_HAZARD = 18
MAP_SEED = 12345

START_ENERGY = 100
ENERGY_PER_STEP = 1
FOOD_ENERGY = 20

# MAX_STEPS is provably unreachable and has never bound. Energy is the real
# constraint: total energy available = START_ENERGY + N_FOOD * FOOD_ENERGY
# = 100 + 18*20 = 460, and each step costs exactly 1, so steps_survived <= 460
# for ANY policy from ANY spawn, regardless of route. Confirmed by
# analyze_max_episode.py, which also shows greedy collects all 18 food from all
# 864 spawns -- so 460 is the exact maximum, not an estimate.
# Absolute maximum fitness = 460 + 18*50 = 1360.
# Revisit this if N_FOOD, FOOD_ENERGY, or START_ENERGY change: the ceiling moves.
MAX_STEPS = 500
FOOD_FITNESS_BONUS = 50

VIEW_RADIUS = 2
P_WALL = 9
P_SELF = 5

N_CHANNELS = 4
N_INPUTS = (2 * VIEW_RADIUS + 1) ** 2 * N_CHANNELS
GENOME_LENGTH = N_INPUTS * 4 + 4

POP_SIZE = 100
N_GENERATIONS = 200
INIT_WEIGHT_RANGE = 1.0

TOURNAMENT_K = 3
MUTATION_RATE = 0.10
MUTATION_SIGMA = 0.2
N_ELITES = 1

GA_SEED = 42

HIDDEN_SIZE = 16
MLP_GENOME_LENGTH = N_INPUTS * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * 4 + 4

# Layer-1 weights are scaled down at init. With ~24 active one-hot inputs, drawing
# W1 from +/-1.0 gives hidden pre-activations of std ~2.8, and tanh(2.8)=0.993 --
# nearly every hidden unit would saturate from birth, going flat and insensitive
# to mutation. +/-0.4 puts std ~1.1, inside tanh's responsive band.
MLP_W1_INIT_RANGE = 0.4

# Reward. Return-matched: the UNDISCOUNTED episode return equals the GA fitness
# exactly (steps + 50*food), so RL and GA optimise the identical objective and
# "did RL beat GA" is a well-posed question. A negative per-step term would have
# inverted the survival incentive GA fitness rewards.
R_STEP = 1.0
R_FOOD = 50.0
R_HAZARD = 0.0

# Sparse arm: food only, explicit death penalty, no per-step signal.
R_SPARSE_STEP = 0.0
R_SPARSE_FOOD = 50.0
R_SPARSE_HAZARD = -50.0

# Return-matched rewards imply Q up to ~1360; MSE on O(1e3) errors against O(1)
# tanh activations diverges at any usable lr. Scaling by a positive constant
# leaves the argmax policy and objective ordering exactly unchanged, so this is
# free. With gamma=0.99 this puts Q at O(2).
REWARD_SCALE = 1.0 / 50.0

GAMMA = 0.99
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_FLOOR = 0.05
EPSILON_DECAY_FRACTION = 0.5
REPLAY_CAPACITY = 50_000
REPLAY_WARMUP = 1_000
BATCH_SIZE = 32
TRAIN_EVERY = 4
TARGET_SYNC_STEPS = 1_000
HUBER_DELTA = 1.0
GRAD_CLIP = 1.0
TOTAL_ENV_STEPS = 500_000
DQN_SEEDS = [1, 2, 3, 4, 5]
