"""
Global config: game constants (from the Kaggriculture README) and RL hyperparameters.
No fertilizer is ever used by this agent, so WHEAT's effective max yield is 4 (not 6).
"""

# ---- Game constants (WHEAT only) ----
WHEAT_SEED_COST = 10
WHEAT_BASE_PRICE = 25
WHEAT_TIME_TO_FIRST_YIELD = 2
WHEAT_TIME_TO_MAX_YIELD = 4          # age at which yield_units caps, without fertilizer
WHEAT_MAX_YIELD_NO_FERTILIZER = 4    # cap on yield_units we ever expect to see
WHEAT_MAX_LIFESPAN_STEP_OFFSET = 1   # one-time crops reach max lifespan one day after max_yield_day

BOARD_SIZE_DEFAULT = 10
QUADRANT_SIZE = 5
LAND_COSTS = [1000, 2000, 4000]      # cost of the 2nd, 3rd, 4th quadrant (1st is free/starting)

EPISODE_STEPS = 720
TURNS_PER_DAY = 24
DAYS = 30

# ---- State encoding ----
LOCAL_WINDOW = 5              # 5x5 window around each unit
LOCAL_CELL_FEATURES = 8       # [locked, empty, is_wheat, age_norm, yield_norm, watered, unwatered_norm, is_weed]
GLOBAL_FEATURES = 16

UNIT_STATE_DIM = GLOBAL_FEATURES + LOCAL_WINDOW * LOCAL_WINDOW * LOCAL_CELL_FEATURES + 1 + 4
# +1 = unit's own wheat inventory (normalized)
# +4 = normalized (dx, dy) to nearest unwatered wheat and nearest harvestable wheat

MARKET_STATE_DIM = GLOBAL_FEATURES

# ---- Action spaces ----
UNIT_ACTIONS = ["NORTH", "SOUTH", "EAST", "WEST", "PLANT", "WATER", "HARVEST", "PICKUP", "DROP", "DIG", "PASS"]
N_UNIT_ACTIONS = len(UNIT_ACTIONS)

# Market action bins: quantities kept small & discrete on purpose (DQN-friendly).
MARKET_ACTIONS = [
    ("NOOP", None),
    ("BUY_SEED", 1),
    ("BUY_SEED", 5),
    ("BUY_SEED", 10),
    ("SELL", 0.25),   # fraction of current shed WHEAT
    ("SELL", 0.5),
    ("SELL", 1.0),
    ("HIRE", None),
    ("BUY_LAND", None),
]
N_MARKET_ACTIONS = len(MARKET_ACTIONS)

# ---- Reward shaping ----
WEED_PENALTY = -2.0
INVALID_ACTION_PENALTY = -1.0
REWARD_MONEY_SCALE = 1.0 / 100.0   # keep reward magnitude reasonable for the network

# ---- RL hyperparameters ----
GAMMA = 0.995
LR = 2.5e-4
BATCH_SIZE = 128
REPLAY_CAPACITY = 200_000
WARMUP_STEPS = 5_000
TARGET_UPDATE_FREQ = 2_000          # hard update every N gradient steps
GRAD_CLIP_NORM = 10.0

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 300_000           # ~ a few hundred episodes of 720 steps each

PER_ALPHA = 0.6                     # prioritization exponent
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_STEPS = 300_000
PER_EPS = 1e-5

NUM_EPISODES = 2000
LOG_EVERY = 10
CHECKPOINT_EVERY = 100
CHECKPOINT_DIR = "checkpoints"
