data_dir = '/vol/grid-solar/sgeusers/nguyenhoai2/Dataset/'

# evolutionary parameters
max_iterations = 100
pop_size = 100

threshold = 0.7

alg_style = 'embed' # embed|wrapper|filter
parallel = False # True: run parallely/online, False: run sequentially/offline

# parameters for wrapper
w_wrapper = 1.0
no_inner_folds = 3

# parameters for Embedded feature selection
# initialization style, can be random or interval initialization
init_style = 'interval'
# if normalize is true, need to maintain
fit_normalized = True
min_reg = -1.0
max_reg = -1.0
min_loss = -1.0
max_loss = -1.0
# parameter for embed feature selection
# fitness = reg + alpha * loss
alpha = 100.0
loss = 'H' # can be binary (B) or Hingle (H) loss
reg = 'l1' # can be l0, l1, l2

# Parameters for filter feature selection
f_measure = 'relief' # relief|cor

